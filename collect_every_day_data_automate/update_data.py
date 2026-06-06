import os
# --- FIX 1: MAC GPU DEADLOCK PREVENTION ---
# Must be set before TensorFlow is imported!
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the absolute paths (since they are in the same folder as the script)
MODEL_PATH = os.path.join(SCRIPT_DIR, "quant_lstm_model.keras")
SCALER_PATH = os.path.join(SCRIPT_DIR, "quant_scaler.pkl")
# Do the same for feature_columns if you load them here:
FEATURES_PATH = os.path.join(SCRIPT_DIR, "feature_columns.pkl")

# Disable visible GPUs in TF just to be completely safe on Mac
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

def build_reconstructed_model():
    reg = l1_l2(l1=1e-5, l2=1e-4)
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=(10, 34), 
             kernel_regularizer=reg, recurrent_regularizer=reg),
        BatchNormalization(),
        Dropout(0.388),
        LSTM(units=64, return_sequences=False, 
             kernel_regularizer=reg, recurrent_regularizer=reg),
        BatchNormalization(),
        Dropout(0.206),
        Dense(8, activation='relu', kernel_regularizer=reg),
        Dense(3, activation='softmax')
    ])
    return model

# --- LOAD LOGIC ---
try:
    print("Reconstructing model architecture...")
    model = build_reconstructed_model()
    model.load_weights(MODEL_PATH) 
    print("✅ Model weights loaded successfully!")
except Exception as e:
    print(f"❌ Critical error loading model: {e}")

# LOAD ARTIFACTS
scaler = joblib.load(SCALER_PATH)
FEATURE_COLS = joblib.load(FEATURES_PATH)
TIME_STEPS = 10

def calculate_local_features(df):
    """Calculates all stock-specific indicators."""
    df = df.copy()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Dist_to_MA50'] = df['Close'] / df['MA_50'] - 1
    df['Dist_to_MA200'] = df['Close'] / df['MA_200'] - 1
    df['Golden_Cross'] = (df['MA_50'] > df['MA_200']).astype(int)
    df['ma_ratio'] = df['MA_50'] / df['MA_200']
    df['volatility_5'] = df['Close'].rolling(5).std()
    df['volatility_20'] = df['Close'].rolling(20).std()

    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['Daily_Return'] = df['Close'].pct_change()
    df['Momentum_Acceleration'] = df['Daily_Return'].diff()
    df['VPT'] = df['Daily_Return'] * df['Volume']
    df['VPT_MA10'] = df['VPT'].rolling(10).mean()
    df['MACD'] = df['MA_200'] - df['MA_50']
    df['MACD_norm'] = df['MACD'] / df['MA_50']

    for w in [5, 10, 20]:
        df[f"close_lag_{w}"] = df['Close'].shift(w)
        df[f"return_lag_{w}"] = df['Daily_Return'].shift(w)
        df[f"volume_lag_{w}"] = df['Volume'].shift(w)
        df[f"close_mean_{w}"] = df['Close'].rolling(w).mean()
        df[f"close_std_{w}"] = df['Close'].rolling(w).std()
        df[f"volume_mean_{w}"] = df['Volume'].rolling(w).mean()
        df[f"return_std_{w}"] = df['Daily_Return'].rolling(w).std()

    df["volume_ratio"] = df["Volume"] / df["volume_mean_20"]

    log_HL = np.log(df['High'] / df['Low'])
    log_CO = np.log(df['Close'] / df['Open'])
    df['GK_Volatility'] = np.sqrt(0.5 * log_HL**2 - (2 * np.log(2) - 1) * log_CO**2)

    df['CLV'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)

    return df.dropna()

def update_daily_data():
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
        "ITC.NS", "LT.NS", "BHARTIARTL.NS", "SBIN.NS", "HINDUNILVR.NS", 
        "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "BAJFINANCE.NS", 
        "NTPC.NS", "ONGC.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "WIPRO.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS", "POWERGRID.NS", 
        "BPCL.NS", "GRASIM.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "INDUSINDBK.NS", 
        "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", 
        "TATACONSUM.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "TECHM.NS",
        "FEDERALBNK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "IOB.NS", 
        "UNIONBANK.NS", "CHOLAFIN.NS", "M&MFIN.NS", "PFC.NS", "LICHSGFIN.NS",
        "PERSISTENT.NS", "MPHASIS.NS", "OFSS.NS", "TATAELXSI.NS", "ASHOKLEY.NS", 
        "TVSMOTOR.NS", "ESCORTS.NS", "BHARATFORG.NS", "BOSCHLTD.NS", "MRF.NS",
        "HINDALCO.NS", "NATIONALUM.NS", "NMDC.NS", "SAIL.NS", "VEDL.NS", 
        "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "RAMCOCEM.NS", "DLF.NS",
        "BHEL.NS", "BEL.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS", 
        "HAVELLS.NS", "TATAPOWER.NS", "JSWENERGY.NS", "NHPC.NS", "TORNTPOWER.NS",
        "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "VOLTAS.NS", 
        "BATAINDIA.NS", "PAGEIND.NS", "TRENT.NS", "JUBLFOOD.NS", "ZEEL.NS",
        "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS", "GLENMARK.NS", "BIOCON.NS", 
        "TATACHEM.NS", "COROMANDEL.NS", "PIDILITIND.NS", "SRF.NS", "DEEPAKNTR.NS"
    ]  

    print("\n--- STEP 1: BULK DOWNLOADING ALL MARKET DATA ---")
    # FIX 2: Download everything simultaneously in a fraction of the time
    bulk_data = yf.download(tickers, period="1y", progress=False)
    
    all_market_data = []

    print("--- STEP 2: PROCESSING LOCAL FEATURES ---")
    for ticker in tickers:
        try:
            # Extract single ticker data from the bulk MultiIndex dataframe
            if isinstance(bulk_data.columns, pd.MultiIndex):
                # Ensure the ticker actually downloaded
                if ticker not in bulk_data.columns.get_level_values(1):
                    continue
                df = bulk_data.xs(ticker, level=1, axis=1).copy()
            else:
                df = bulk_data.copy()

            if df.empty or df['Close'].isna().all():
                continue
                
            df = df.ffill().bfill()
            
            # Feature engineering
            df = calculate_local_features(df)
            df['Ticker'] = ticker  
            all_market_data.append(df)
            
        except Exception as e:
            pass # Silently skip bad data

    if not all_market_data:
        print(" No valid market data processed. Exiting.")
        return

    # Combine all individual stocks
    master_df = pd.concat(all_market_data)

    print("--- STEP 3: CALCULATING MARKET-WIDE CROSS-RANKS ---")
    cross_sectional_features = ['Daily_Return', 'RSI_14', 'Dist_to_MA50', 'GK_Volatility']
    for col in cross_sectional_features:
        if col in master_df.columns:
            master_df[f'{col}_CrossRank'] = master_df.groupby(master_df.index)[col].rank(pct=True)


    # DEBUG: Save the raw, unscaled features to Excel/CSV to manually check them
    # master_df[master_df['Ticker'] == 'RELIANCE.NS'].tail(5).to_csv("DEBUG_RELIANCE.csv")

    print("--- STEP 4: PREPARING BATCH TENSORS ---")
    batch_inputs = []
    valid_tickers = []
    
    for ticker in tickers:
        try:
            ticker_df = master_df[master_df['Ticker'] == ticker].copy()
            if ticker_df.empty:
                continue

            for col in FEATURE_COLS:
                if col not in ticker_df.columns:
                    ticker_df[col] = 0.0

            ticker_df = ticker_df[FEATURE_COLS]
            scaled_data = scaler.transform(ticker_df)

            if len(scaled_data) < TIME_STEPS:
                continue

            # Extract the last 10 days
            last_10 = scaled_data[-TIME_STEPS:]
            batch_inputs.append(last_10)
            valid_tickers.append(ticker)
            
        except Exception:
            continue

    print(f"--- STEP 5: RUNNING INSTANT BATCH INFERENCE ({len(batch_inputs)} Stocks) ---")
    predictions = {}
    
    if len(batch_inputs) > 0:
        X_batch = np.array(batch_inputs) 
        
        # FIX 3: DIRECT TENSOR MATH. Bypasses buggy predict() loop
        batch_preds = model(X_batch, training=False).numpy()
        
        for i, ticker in enumerate(valid_tickers):
            pred = batch_preds[i]
            predictions[ticker] = {
                "Sell": float(pred[0]),
                "Hold": float(pred[1]),
                "Buy":  float(pred[2])
            }

    # SAVE OUTPUT
    os.makedirs("data", exist_ok=True)
    with open("data/latest_predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    print("Night inference completed successfully in record time!")

if __name__ == "__main__":
    update_daily_data()
