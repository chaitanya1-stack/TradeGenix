import pandas as pd
import json
import yfinance as yf
import difflib
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import math



# =====================================================================
# 1. API INITIALIZATION
# =====================================================================
app = FastAPI(title="Quant Trading Dashboard API", description="Provides AI predictions, top 5 stocks, and live chart data.")

# Enable CORS so your React/Next.js frontend can communicate with this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMPANY_DIRECTORY = {}  
TICKER_TO_NAME = {}     

def load_master_directory():
    """Loads the CSV file to map company names like 'TCS' to 'TCS.NS'"""
    global COMPANY_DIRECTORY, TICKER_TO_NAME
    try:
        # Update this path if your CSV is in a different folder
        df = pd.read_csv("contains_all_ticker_with_name.csv") 
        for _, row in df.iterrows():
            name = str(row['NAME OF COMPANY']).strip()
            ticker = str(row['SYMBOL']).strip().upper()
            if not ticker.endswith(".NS"):
                ticker += ".NS"
            
            COMPANY_DIRECTORY[name.lower()] = ticker
            TICKER_TO_NAME[ticker] = name
        print(f"✅ Loaded {len(COMPANY_DIRECTORY)} companies into Search Database.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load contains_all_ticker_with_name.csv: {e}")

load_master_directory()

# =====================================================================
# 2. CORE LOGIC
# =====================================================================
def get_ai_prediction(ticker: str):
    """Reads the nightly batch JSON to get the AI's latest opinion."""
    try:
        with open("collect_every_day_data_automate/data/latest_predictions.json", "r") as f:
            predictions = json.load(f)
        return predictions.get(ticker, None)
    except FileNotFoundError:
        return None

# =====================================================================
# 3. API ENDPOINTS
# =====================================================================

@app.get("/search")
def search(query: str):
    """Autocorrect & Fuzzy Search Endpoint"""
    q = query.lower().strip()
    
    # 1. Look for exact matches or substrings
    matches = [
        {"name": name.title(), "ticker": ticker} 
        for name, ticker in COMPANY_DIRECTORY.items() 
        if q in name or q in ticker.lower()
    ]
    
    # 2. If no exact match, use difflib to guess what they meant (Autocorrect)
    if not matches:
        fuzzy_names = difflib.get_close_matches(q, list(COMPANY_DIRECTORY.keys()), n=5, cutoff=0.4)
        matches = [{"name": m.title(), "ticker": COMPANY_DIRECTORY[m]} for m in fuzzy_names]
        
    return {"suggestions": matches[:10]}


@app.get("/top5")
def top_5_stocks():
    """Returns the top 5 stocks with the highest AI 'Buy' probability."""
    try:
        with open("collect_every_day_data_automate/data/latest_predictions.json", "r") as f:
            predictions = json.load(f)
            
        # Sort all stocks by their 'Buy' probability in descending order
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1]['Buy'], reverse=True)
        top_5 = sorted_preds[:5]
        
        result = []
        for ticker, probs in top_5:
            try:
                # Fetch a 5-day snapshot and drop NaNs to prevent crashes
                tk = yf.Ticker(ticker)
                fast_hist = tk.history(period="5d").dropna(subset=['Close'])
                
                day_change = 0.0
                last_price = 0.0
                
                if len(fast_hist) >= 2:
                    last_price = float(fast_hist['Close'].iloc[-1])
                    prev_close = float(fast_hist['Close'].iloc[-2])
                    
                    # Prevent division by zero
                    if prev_close > 0:
                        day_change = ((last_price - prev_close) / prev_close) * 100
                        
                elif len(fast_hist) == 1:
                    last_price = float(fast_hist['Close'].iloc[-1])
                    
                result.append({
                    "ticker": ticker,
                    "company_name": TICKER_TO_NAME.get(ticker, ticker.replace(".NS", "")),
                    "ai_signals": {
                        "buy_probability": round(probs['Buy'] * 100, 2),
                        "hold_probability": round(probs['Hold'] * 100, 2),
                        "sell_probability": round(probs['Sell'] * 100, 2)
                    },
                    "live_stats": {
                        "last_price": round(last_price, 2),
                        "day_change_pct": round(day_change, 2)
                    }
                })
            except Exception as e:
                print(f"Skipping {ticker} for Top 5 due to data error: {e}")
                continue # Skip the broken stock and move to the next one
            
        return {"success": True, "top_5": result}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="AI predictions file not found. Run update_data.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{ticker_or_name}")
def predict(ticker_or_name: str):
    """The main endpoint: Returns AI data, OHLCV stats, and multi-timeframe charts."""
    query = ticker_or_name.lower().strip()
    
    # 1. Resolve User Input (Autocorrect name to ticker)
    ticker = COMPANY_DIRECTORY.get(query)
    if not ticker:
        ticker = query.upper()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
            
    # 2. Get AI Prediction from JSON
    ai_data = get_ai_prediction(ticker)
    if not ai_data:
        ai_data = {"Sell": 0.0, "Hold": 0.0, "Buy": 0.0, "Note": "AI Prediction not available."}
        action = "N/A"
    else:
        # Determine the highest probability action
        action_idx = np.argmax([ai_data["Sell"], ai_data["Hold"], ai_data["Buy"]])
        action = ["Sell", "Hold", "Buy"][action_idx]

    # 3. Fetch Live Market Data & Charts from Yahoo Finance
    try:
        tk = yf.Ticker(ticker)
        
        # Get full month of data, DROP any rows where Close is NaN
        hist_1mo = tk.history(period="1mo", interval="1d").dropna(subset=['Close'])
        if hist_1mo.empty:
            raise ValueError("No market data found for this ticker.")
            
        latest_bar = hist_1mo.iloc[-1]
        prev_bar = hist_1mo.iloc[-2] if len(hist_1mo) > 1 else latest_bar
        
        # Prevent division by zero (Infinity error)
        prev_close = float(prev_bar['Close'])
        if prev_close > 0:
            day_change = ((float(latest_bar['Close']) - prev_close) / prev_close) * 100
        else:
            day_change = 0.0

        stats = {
            "Open": round(float(latest_bar['Open']), 2),
            "High": round(float(latest_bar['High']), 2),
            "Low": round(float(latest_bar['Low']), 2),
            "Close": round(float(latest_bar['Close']), 2),
            "Volume": int(latest_bar['Volume']),
            "Previous_Close": round(prev_close, 2),
            "Day_Change_Pct": round(day_change, 2)
        }

        # Generate Chart Arrays (Ensure we drop NaNs so JSON doesn't crash)
        hist_1d = tk.history(period="1d", interval="5m").dropna(subset=['Close'])
        hist_1w = tk.history(period="5d", interval="15m").dropna(subset=['Close'])

        # Create chart dictionaries, ensuring no NaNs sneak in
        chart_1d = [{"time": str(idx.strftime('%H:%M')), "price": round(float(row['Close']), 2)} for idx, row in hist_1d.iterrows()]
        chart_1w = [{"time": str(idx.strftime('%a %H:%M')), "price": round(float(row['Close']), 2)} for idx, row in hist_1w.iterrows()]
        chart_1m = [{"time": str(idx.strftime('%Y-%m-%d')), "price": round(float(row['Close']), 2)} for idx, row in hist_1mo.iterrows()]

        return {
            "success": True,
            "company_name": TICKER_TO_NAME.get(ticker, ticker.replace(".NS", "")),
            "ticker": ticker,
            "prediction": {
                "action": action,
                "probabilities": {
                    "Sell": round(ai_data["Sell"] * 100, 2),
                    "Hold": round(ai_data["Hold"] * 100, 2),
                    "Buy": round(ai_data["Buy"] * 100, 2)
                }
            },
            "live_stats": stats,
            "charts": {
                "1D": chart_1d,
                "1W": chart_1w,
                "1M": chart_1m
            }
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch market data: {str(e)}")