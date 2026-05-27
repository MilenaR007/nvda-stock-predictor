import os
import warnings
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from alpaca_trade_api.rest import REST
from transformers import pipeline as hf_pipeline
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NVIDIA Stock Predictor", layout="wide")
st.title("📈 NVIDIA Stock Predictor")
st.write("This application analyzes news sentiment (Alpaca + FinBERT) and technical data (Yahoo Finance) to predict the direction of NVDA stock price.")

# --- LOAD SENTIMENT MODEL (Cached for speed) ---
@st.cache_resource
def load_finbert():
    return hf_pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", top_k=None)

_finbert_pipeline = load_finbert()

# --- HELPER FUNCTIONS ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd_signal(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

def compute_rolling_beta(asset_ret, market_ret, window=21):
    covariance = asset_ret.rolling(window).cov(market_ret)
    variance = market_ret.rolling(window).var()
    return covariance / variance

def fetch_news_sentiment(symbol, start, end, trading_days):
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        st.warning("Alpaca API keys missing in 'Secrets'. News sentiment will be skipped (zero-filled).")
        return pd.DataFrame()

    api = REST(api_key, api_secret, base_url="https://paper-api.alpaca.markets", api_version="v2", raw_data=False)
    api._retry = 1 
    
    all_headlines = []
    all_timestamps = []
    
    try:
        while True:
            kwargs = {"symbol": symbol, "start": start, "end": end, "limit": 50}
            news = api.get_news(**kwargs)
            if not news: break
            for article in news:
                all_headlines.append(article.headline)
                all_timestamps.append(article.created_at)
            if len(news) < 50: break
            last_ts = news[-1].created_at
            if last_ts: end = pd.Timestamp(last_ts).isoformat()
            else: break
    except Exception as e:
        return pd.DataFrame()

    if not all_headlines:
        return pd.DataFrame()
        
    compounds = []
    for batch_start in range(0, len(all_headlines), 32):
        batch = all_headlines[batch_start : batch_start + 32]
        batch_results = _finbert_pipeline(batch, truncation=True, max_length=512)
        for result in batch_results:
            scores = {r["label"]: r["score"] for r in result}
            compound_score = scores.get("positive", 0) - scores.get("negative", 0)
            compounds.append(compound_score)

    market_close = time(16, 0)
    trade_dates = []
    for ts_raw in all_timestamps:
        ts = pd.Timestamp(ts_raw)
        ts = ts.tz_localize("America/New_York") if ts.tzinfo is None else ts.tz_convert("America/New_York")
        cal_date = (ts + timedelta(days=1)).normalize().tz_localize(None) if ts.time() >= market_close else ts.normalize().tz_localize(None)
        valid = trading_days[trading_days >= cal_date]
        trade_dates.append(valid[0] if len(valid) > 0 else cal_date)
        
    articles_df = pd.DataFrame({"date": trade_dates, "compound": compounds})
    daily = articles_df.groupby("date").agg(
        sent_mean=("compound", "mean"), sent_std=("compound", "std"),
        sent_max=("compound", "max"), sent_min=("compound", "min"),
        news_count=("compound", "count"),
    )
    daily["sent_std"] = daily["sent_std"].fillna(0)
    daily.index.name = None
    return daily

def prepare_data(start_date, end_date):
    df = yf.download("NVDA", start=start_date, end=end_date, progress=False)
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    
    df.columns = [col[0].lower() for col in df.columns]
    df = df.rename(columns={'close': 'nvda_close', 'high': 'nvda_high', 'low': 'nvda_low', 'volume': 'nvda_volume'})
    spy_close = spy['Close'].squeeze()
    
    nvda_log_ret = np.log(df["nvda_close"] / df["nvda_close"].shift(1))
    spy_log_ret = np.log(spy_close / spy_close.shift(1))
    
    feat = pd.DataFrame(index=df.index)
    for lag in [1, 2, 3, 5, 10, 21]: feat[f"nvda_ret_lag{lag}"] = nvda_log_ret.shift(lag)
    for w in [5, 10, 21]:
        feat[f"nvda_vol_{w}d"] = nvda_log_ret.rolling(w).std()
        feat[f"nvda_mean_ret_{w}d"] = nvda_log_ret.rolling(w).mean()
        feat[f"spy_vol_{w}d"] = spy_log_ret.rolling(w).std()
        
    feat["nvda_rsi_14"] = compute_rsi(df["nvda_close"], 14)
    feat["nvda_macd_hist"] = compute_macd_signal(df["nvda_close"])
    feat["nvda_vol_ratio_5_21"] = df["nvda_volume"].rolling(5).mean() / df["nvda_volume"].rolling(21).mean()
    feat["nvda_volume_chg_1d"] = df["nvda_volume"].pct_change(1)
    
    ma20 = df["nvda_close"].rolling(20).mean()
    std20 = df["nvda_close"].rolling(20).std()
    feat["nvda_bb_width"] = (2 * std20) / ma20
    feat["nvda_hl_range"] = (df["nvda_high"] - df["nvda_low"]) / df["nvda_close"]
    feat["spy_ret_1d"] = spy_log_ret
    feat["spy_ret_5d"] = spy_log_ret.rolling(5).sum()
    feat["idio_ret_1d"] = nvda_log_ret - spy_log_ret
    feat["nvda_beta_spy_21d"] = compute_rolling_beta(nvda_log_ret, spy_log_ret, 21)
    
    trading_days = df.index.tz_localize(None)
    sentiment = fetch_news_sentiment("NVDA", start=start_date, end=end_date, trading_days=trading_days)
    
    if not sentiment.empty:
        feat = feat.join(sentiment, how="left")
        feat.fillna({'sent_mean': 0, 'sent_std': 0, 'sent_max': 0, 'sent_min': 0, 'news_count': 0}, inplace=True)
    else:
        for col in ['sent_mean', 'sent_std', 'sent_max', 'sent_min', 'news_count']: feat[col] = 0.0
            
    feat['target'] = nvda_log_ret.shift(-1)
    return feat.dropna()

def train_and_evaluate_models(data):
    X = data.drop(columns=['target'])
    y = data['target']
    
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_preds = ridge_model.predict(X_test_scaled)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1}
    
    gbm_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_eval], callbacks=[lgb.log_evaluation(0)])
    lgb_preds = gbm_model.predict(X_test)
    
    return y_test, ridge_preds, lgb_preds

# --- MAIN INTERFACE ---
if st.button("Run Analysis", type="primary"):
    with st.spinner("Fetching market data, news, and training models... This may take a moment."):
        START_DATE = "2023-01-01"
        END_DATE = datetime.today().strftime('%Y-%m-%d')
        
        dataset = prepare_data(START_DATE, END_DATE)
        
        if dataset.empty:
            st.error("Failed to fetch data. Please try again later.")
        else:
            y_test, ridge_preds, lgb_preds = train_and_evaluate_models(dataset)
            st.success("Models trained successfully!")
            
            # --- CHART ---
            st.subheader("📊 Actual vs. Predicted Log-Returns (Test Set)")
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(y_test.index, y_test.values, label="Actual Return", color='black', linewidth=1.5, alpha=0.8)
            ax.plot(y_test.index, ridge_preds, label="Ridge Regression", color='blue', linewidth=1.5, alpha=0.7)
            ax.plot(y_test.index, lgb_preds, label="LightGBM", color='darkorange', linewidth=1.5, alpha=0.7)
            
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.legend(loc="upper left")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)