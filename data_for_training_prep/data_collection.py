import yfinance as yf
import pandas as pd
from tqdm import tqdm

tickers = [
    #  TIER 1: The Nifty 50 Anchors (40 Stocks) 
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
    "ITC.NS", "LT.NS", "BHARTIARTL.NS", "SBIN.NS", "HINDUNILVR.NS", 
    "M&M.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "BAJFINANCE.NS", 
    "NTPC.NS", "ONGC.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS", "POWERGRID.NS", 
    "BPCL.NS", "GRASIM.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "INDUSINDBK.NS", 
    "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", 
    "TATACONSUM.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "TECHM.NS",

    #  TIER 2: Legacy Banking & Financials (10 Stocks) 
    "FEDERALBNK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "IOB.NS", 
    "UNIONBANK.NS", "CHOLAFIN.NS", "M&MFIN.NS", "PFC.NS", "LICHSGFIN.NS",

    #  TIER 3: Legacy Auto & IT Mid-Caps (10 Stocks) 
    "PERSISTENT.NS", "MPHASIS.NS", "OFSS.NS", "TATAELXSI.NS", "ASHOKLEY.NS", 
    "TVSMOTOR.NS", "ESCORTS.NS", "BHARATFORG.NS", "BOSCHLTD.NS", "MRF.NS",

    #  TIER 4: Metals, Mining & Real Estate (10 Stocks)
    "HINDALCO.NS", "NATIONALUM.NS", "NMDC.NS", "SAIL.NS", "VEDL.NS", 
    "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "RAMCOCEM.NS", "DLF.NS",

    #  TIER 5: Power, Defense & Capital Goods (10 Stocks) 
    "BHEL.NS", "BEL.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS", 
    "HAVELLS.NS", "TATAPOWER.NS", "JSWENERGY.NS", "NHPC.NS", "TORNTPOWER.NS",

    #  TIER 6: Deep History FMCG & Consumer (10 Stocks) 
    "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "VOLTAS.NS", 
    "BATAINDIA.NS", "PAGEIND.NS", "TRENT.NS", "JUBLFOOD.NS", "ZEEL.NS",

    #  TIER 7: Deep History Pharma & Chemicals (10 Stocks)
    "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS", "GLENMARK.NS", "BIOCON.NS", 
    "TATACHEM.NS", "COROMANDEL.NS", "PIDILITIND.NS", "SRF.NS", "DEEPAKNTR.NS"
]

all_data = []

for ticker in tqdm(tickers, desc="Downloading stock data"):
    start_date = "2011-01-01"
    end_date = "2026-01-01"
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print(f"No data for {ticker}")
        continue

    #  Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    df['Ticker'] = ticker

    all_data.append(df)

# Combine everything
final_df = pd.concat(all_data, ignore_index=True)

#  Reorder columns (clean structure)
final_df = final_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]

# Save
final_df.to_csv("stock_data.csv", index=False)

print(" Clean data saved!")
print(final_df.head())