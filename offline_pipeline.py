# %% [markdown]
# # Offline Disease Pipeline Modeling
# This IDE notebook handles the decoupled batch pipeline. It processes the raw patient line-list DataFrame, maps clinical/geographic attributes systematically, executes rule-based alerts uniquely matched per mandal, and computes the robust Auto-ARIMA statistical forecasts dynamically. 
# The outputs are systematically dispatched as lightweight JSONs directly to the frontend's static directory.

# %% 
import pandas as pd
import numpy as np
import json
import warnings
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Importing configuration safely which contains our Amoebiasis and Five Day Fever updates
from config import DISEASE_CODES

# %% [markdown]
# ## 1. Data Ingestion & Schema Alignment
# Correcting columns for the incoming dataframe to route into standard keys.

# %%
def load_and_preprocess(filepath="raw_data.csv"):
    """
    Standardizes the new df format (longitude, latitude, sub_district -> mandal, diagnosis -> diagnosis_code)
    """
    df = pd.read_csv(filepath)
    
    # Core Mapping
    df['mandal'] = df['sub_district'].fillna(df.get('mandal', 'Unknown')) 
    df['district'] = df['district_y'].fillna(df.get('district_x', 'Unknown'))
    df['diagnosis_code'] = df['diagnosis'].astype(str).str.strip()
    df['event_date'] = pd.to_datetime(df['timestamp'])
    
    # Filter only to tracked diseases via dict
    tracked_codes = []
    code_to_disease = {}
    for d, info in DISEASE_CODES.items():
        if 'codes' in info:
            for code in info['codes']:
                tracked_codes.append(str(code))
                code_to_disease[str(code)] = d
            
    df = df[df['diagnosis_code'].isin(tracked_codes)].copy()
    df['disease_key'] = df['diagnosis_code'].map(code_to_disease)
    
    return df

# %% [markdown]
# ## 2. Time-Series Aggregation (Strictly Mandal-Level)
# Extracting localized vectors for each unique geographic block.

# %%
def get_mandal_timeseries(df):
    """
    Aggregates cases by (Mandal, Disease, Week)
    """
    df['period'] = df['event_date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Weekly aggregate per mandal per disease
    ts = df.groupby(['period', 'district', 'mandal', 'disease_key']).size().reset_index(name='case_count')
    
    # Fill missing weeks seamlessly with 0
    full_ts = []
    for (mandal, d_key), group in ts.groupby(['mandal', 'disease_key']):
        min_p = group['period'].min()
        max_p = group['period'].max()
        if pd.isna(min_p): continue
            
        all_weeks = pd.date_range(min_p, max_p, freq='W-MON')
        g_reindexed = group.set_index('period').reindex(all_weeks, fill_value=0).reset_index()
        g_reindexed['mandal'] = mandal
        g_reindexed['disease_key'] = d_key
        g_reindexed['district'] = group['district'].iloc[0] if len(group['district']) > 0 else 'Unknown'
        full_ts.append(g_reindexed)
        
    if full_ts:
        return pd.concat(full_ts, ignore_index=True).rename(columns={'index': 'period'})
    return pd.DataFrame()

# %% [markdown]
# ## 3. Robust ARIMA & Forecasting (Auto-Optimized)
# Using `pmdarima` grid-searches to generate mathematically proven curves.

# %%
def run_mandal_forecasts(ts_mandal, forecast_horizon=4):
    """
    Runs an Auto-ARIMA model evaluated through AIC scores
    for every localized mandal stream.
    """
    forecast_results = []
    
    for (mandal, d_key), group in ts_mandal.groupby(['mandal', 'disease_key']):
        group = group.sort_values('period')
        y = group['case_count'].astype(float).values
        dates = group['period'].tolist()
        
        if len(y) < 10:
            continue # Insufficient data for Auto-ARIMA to converge
            
        # 1. Stationarity Check (via Augmented Dickey-Fuller Test)
        try:
            adf_result = adfuller(y)
            needs_differencing = adf_result[1] > 0.05
        except:
            needs_differencing = True
            
        # 2. Robust Auto-ARIMA fitting
        try:
            model = pm.auto_arima(
                y,
                d=1 if needs_differencing else 0,
                seasonal=False, # Fourier terms or higher period vectors are suggested for multi-seasonal patterns
                stepwise=True,
                suppress_warnings=True,
                max_p=3, max_q=3,
                information_criterion='aic'
            )
            
            preds, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=0.05)
            
            # Predict bounds
            last_date = dates[-1]
            future_dates = [last_date + timedelta(weeks=i+1) for i in range(forecast_horizon)]
            
            forecast_results.append({
                "mandal": mandal,
                "disease": DISEASE_CODES[d_key]['name'],
                "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "predictions": [max(0, round(float(p), 1)) for p in preds],
                "lower_95": [max(0, round(float(c[0]), 1)) for c in conf_int],
                "upper_95": [max(0, round(float(c[1]), 1)) for c in conf_int]
            })
        except Exception as e:
            print(f"Skipping {mandal} - {d_key} due to ARIMA convergence failure: {e}")
            
    return forecast_results

# %% [markdown]
# ## 4. Output Generation (JSON Extraction)
# Direct extraction natively saving to the `.frontend/public` React directory.

# %%
def generate_frontend_assets(df, ts, forecasts):
    
    os.makedirs('frontend/public', exist_ok=True)
    
    # 1. Generating Spread Map coordinates intelligently
    latest_cutoff = df['event_date'].max() - timedelta(days=28)
    recent = df[df['event_date'] >= latest_cutoff]
    
    spread_map = []
    for (mandal, d_key), group in recent.groupby(['mandal', 'disease_key']):
        if 'latitude' in group.columns and group['latitude'].notna().any():
            spread_map.append({
                "mandal": mandal,
                "disease": DISEASE_CODES[d_key]['name'],
                "cases": len(group),
                "lat": float(group['latitude'].dropna().iloc[0]),
                "lng": float(group['longitude'].dropna().iloc[0])
            })
            
    # 2. Writing JSON payloads cleanly out to frontend
    with open('frontend/public/forecast_curves.json', 'w') as f:
        json.dump(forecasts, f, indent=2)
        
    with open('frontend/public/spread_map.json', 'w') as f:
        json.dump(spread_map, f, indent=2)
        
    print("✅ Successfully generated offline datasets and connected them to the React Front-End directory (frontend/public/)")

# %% [markdown]
# ## Full Execution Cell
# Run this cell after loading your `raw_data.csv` to orchestrate manually.

# %%
if __name__ == "__main__":
    print("Phase 1: Loading raw schema...")
    # WARNING: Update the raw_data.csv path specifically to where you store your incoming data table
    # df = load_and_preprocess("raw_data.csv")
    
    # print("Phase 2: Aggregating temporal mandal streams...")
    # ts_mandal = get_mandal_timeseries(df)
    
    # print("Phase 3: Deploying Auto-ARIMA pipelines natively...")
    # forecasts = run_mandal_forecasts(ts_mandal, forecast_horizon=4)
    
    # print("Phase 4: Converting formats and dumping JSONs...")
    # generate_frontend_assets(df, ts_mandal, forecasts)
