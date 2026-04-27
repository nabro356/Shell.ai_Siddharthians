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
# ## 3. Dynamic Multi-Model Forecasting (The Racing Engine)
# Competes ARIMA, XGBoost, Holt-Winters, ETS, and UCM (BSTS-equivalent) evaluating validation holdouts to extract the lowest RMSE per Mandal dynamically.

# %%
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pmdarima.preprocessing import FourierFeaturizer

def extract_xgb_features(y_series, lags=4):
    """Generates autoregressive lag features for the XGBoost supervised mapping."""
    df = pd.DataFrame({'y': y_series})
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df.dropna()

def run_mandal_forecasts(ts_mandal, forecast_horizon=4):
    """
    Races multiple models using a walk-forward holdout validation.
    The model with the best RMSE exclusively generates the predictions for the UI.
    """
    forecast_results = []
    
    for (mandal, d_key), group in ts_mandal.groupby(['mandal', 'disease_key']):
        group = group.sort_values('period')
        y = group['case_count'].astype(float).values
        dates = group['period'].tolist()
        
        # We need sufficient data for a meaningful 4-week holdout test (e.g. at least 15 weeks)
        if len(y) < 15:
            continue
            
        train = y[:-forecast_horizon]
        test = y[-forecast_horizon:]
        
        models_performance = {}
        predictions_future = {}
        
        # --- 1. Auto-ARIMA ---
        try:
            # Stationarity Check mathematically handles the trend differencing parameter (d)
            needs_diff = adfuller(train)[1] > 0.05
            
            # Using Exogenous Fourier Terms for Robust Weekly Seasonality (Annual = 52.14 weeks)
            if len(train) > 52:
                # Extracts 2 pairs of sine/cosine waves representing the 52-week seasonal cycle
                fourier = FourierFeaturizer(m=52.14, k=2)
                train_y, train_X = fourier.fit_transform(train)
                
                # Model uses automated AIC grid search over p, q parameters incorporating Fourier X explicitly
                arima_model = pm.auto_arima(train_y, X=train_X, d=1 if needs_diff else 0, seasonal=False, stepwise=True, suppress_warnings=True)
                
                # Transform needs dummy indices length equal to forecast_horizon to generate the next wave points
                _, test_X = fourier.transform(np.zeros(len(test)))
                arima_preds = arima_model.predict(n_periods=forecast_horizon, X=test_X)
            else:
                # Fallback to plain auto_arima parameter optimization if insufficient data for robust annual seasonality
                arima_model = pm.auto_arima(train, d=1 if needs_diff else 0, seasonal=False, stepwise=True, suppress_warnings=True)
                arima_preds = arima_model.predict(n_periods=forecast_horizon)
                
            models_performance['ARIMA'] = np.sqrt(mean_squared_error(test, arima_preds))
            
            # Refit on full series to generate final Future Predictions
            if len(y) > 52:
                fourier_full = FourierFeaturizer(m=52.14, k=2)
                full_y, full_X = fourier_full.fit_transform(y)
                refit_arima = pm.auto_arima(full_y, X=full_X, d=1 if needs_diff else 0, seasonal=False, stepwise=True, suppress_warnings=True)
                
                _, future_X = fourier_full.transform(np.zeros(forecast_horizon))
                predictions_future['ARIMA'] = refit_arima.predict(n_periods=forecast_horizon, X=future_X)
            else:
                refit_arima = pm.auto_arima(y, d=1 if needs_diff else 0, seasonal=False, stepwise=True, suppress_warnings=True)
                predictions_future['ARIMA'] = refit_arima.predict(n_periods=forecast_horizon)
        except: pass
        
        # --- 2. Holt-Winters (Exponential Smoothing / ETS) ---
        try:
            hw_model = ExponentialSmoothing(train, trend='add', seasonal=None, initialization_method="estimated").fit()
            hw_preds = hw_model.forecast(forecast_horizon)
            models_performance['Holt-Winters'] = np.sqrt(mean_squared_error(test, hw_preds))
            
            refit_hw = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated").fit()
            predictions_future['Holt-Winters'] = refit_hw.forecast(forecast_horizon)
        except: pass

        # --- 3. UCM (Unobserved Components Model / BSTS equivalent) ---
        try:
            ucm_model = UnobservedComponents(train, level='local linear trend').fit(disp=False)
            ucm_preds = ucm_model.forecast(steps=forecast_horizon)
            models_performance['UCM'] = np.sqrt(mean_squared_error(test, ucm_preds))
            
            refit_ucm = UnobservedComponents(y, level='local linear trend').fit(disp=False)
            predictions_future['UCM'] = refit_ucm.forecast(steps=forecast_horizon)
        except: pass
        
        # --- 4. XGBoost Regressor ---
        try:
            feat_df = extract_xgb_features(train, lags=4)
            if not feat_df.empty:
                X_train, Y_train = feat_df.drop(columns=['y']), feat_df['y']
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
                xgb_model.fit(X_train, Y_train)
                
                # Recursive forecast for holdout test set
                xgb_preds = []
                last_lags = train[-4:].tolist()
                for _ in range(forecast_horizon):
                    pred = xgb_model.predict(np.array([last_lags[::-1]]))[0]
                    xgb_preds.append(pred)
                    last_lags.append(pred)
                    last_lags.pop(0)
                models_performance['XGBoost'] = np.sqrt(mean_squared_error(test, xgb_preds))
                
                # Refit blindly on the full series
                feat_full = extract_xgb_features(y, lags=4)
                X_full, Y_full = feat_full.drop(columns=['y']), feat_full['y']
                refit_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
                refit_xgb.fit(X_full, Y_full)
                
                final_xgb_preds = []
                last_lags_f = y[-4:].tolist()
                for _ in range(forecast_horizon):
                    pred = refit_xgb.predict(np.array([last_lags_f[::-1]]))[0]
                    final_xgb_preds.append(pred)
                    last_lags_f.append(pred)
                    last_lags_f.pop(0)
                predictions_future['XGBoost'] = final_xgb_preds
        except: pass

        # --- Dynamic Selection Protocol ---
        if not models_performance:
            continue
            
        # The ultimate winner for this specific mandal is whichever model proved the lowest RMSE mathematically
        best_model_name = min(models_performance, key=models_performance.get)
        best_preds = predictions_future[best_model_name]
        
        last_date = dates[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(forecast_horizon)]
        
        forecast_results.append({
            "mandal": mandal,
            "disease": DISEASE_CODES[d_key]['name'],
            "model_used": best_model_name,
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "predictions": [max(0, round(float(p), 1)) for p in best_preds]
        })
            
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
