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
import glob

def load_and_preprocess(filepaths=None):
    """
    Standardizes the new df format and maps names based on codes globally within the dataset.
    """
    if filepaths is None:
        filepaths = ["raw_data.csv"] # default fallback
        
    dfs = []
    for fp in filepaths:
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp, dtype=str))
    
    if not dfs:
        # Check for matching pattern like 2024.csv, 2025.csv
        csv_files = glob.glob("*202*.csv")
        if csv_files:
            dfs = [pd.read_csv(f, dtype=str) for f in csv_files]
        else:
            raise ValueError("No data files provided or found.")
            
    df = pd.concat(dfs, ignore_index=True)
    
    # Extract codes and names, defaulting if needed
    for col in ["sub_district", "mandal", "district", "district_x", "district_y", "diagnosis", "diagnosis_code"]:
        if col not in df.columns:
            df[col] = np.nan
            
    # 1. Mandal Mapping
    if "sub_district" in df.columns and "mandal" in df.columns:
        sub_is_code = df["sub_district"].astype(str).str.contains(r'\d', na=False).any()
        mandal_is_code = df["mandal"].astype(str).str.contains(r'\d', na=False).any()
        
        if mandal_is_code and not sub_is_code:
            m_code_col = "mandal"
            m_name_col = "sub_district"
            df["mandal"] = df["sub_district"] # Shift string to mandal column for ML
        elif sub_is_code and not mandal_is_code:
            m_code_col = "sub_district"
            m_name_col = "mandal"
        else:
            df["mandal_combined"] = df["sub_district"].fillna(df["mandal"])
            m_name_col = "mandal_combined"
            m_code_col = "sub_district_code" if "sub_district_code" in df.columns else "mandal_code"
    else:
        df["mandal_combined"] = df.get("sub_district", df.get("mandal", pd.Series(dtype=str)))
        m_name_col = "mandal_combined"
        m_code_col = "sub_district_code" if "sub_district_code" in df.columns else "mandal_code"
    
    if m_code_col in df.columns:
        m_map = df.dropna(subset=[m_code_col, m_name_col]).groupby(m_code_col)[m_name_col].first().to_dict()
        df[m_name_col] = df[m_name_col].fillna(df[m_code_col].map(m_map))
    df['mandal'] = df.get(m_name_col, pd.Series(dtype=str)).fillna('Unknown')

    # 2. District Mapping
    if "district_y" in df.columns and "district_x" in df.columns:
        tmp_dist = df["district_y"].fillna(df["district_x"])
        df["district"] = tmp_dist.fillna(df["district"]) if "district" in df.columns else tmp_dist
    
    d_name_col = "district" if "district" in df.columns else "district_y"
    d_code_col = "district_code"
    if d_code_col in df.columns and d_name_col in df.columns:
        d_map = df.dropna(subset=[d_code_col, d_name_col]).groupby(d_code_col)[d_name_col].first().to_dict()
        df[d_name_col] = df[d_name_col].fillna(df[d_code_col].map(d_map))
    df['district'] = df.get(d_name_col, pd.Series(dtype=str)).fillna('Unknown')
        
    # 3. Diagnosis Mapping
    diag_code_col = "diagnosis" if "diagnosis" in df.columns else "diagnosis_code"
    diag_name_col = "diagnosis_name"
    if diag_code_col in df.columns and diag_name_col in df.columns:
        # Map using the most frequent name for that code
        diag_map = df.dropna(subset=[diag_code_col, diag_name_col]).groupby(diag_code_col)[diag_name_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).to_dict()
        df[diag_name_col] = df[diag_name_col].fillna(df[diag_code_col].map(diag_map))
        
    df['diagnosis_code'] = df.get(diag_code_col, pd.Series(dtype=str)).astype(str).str.strip()
    df['event_date'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
    
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
    if df is None or df.empty:
        return pd.DataFrame()
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
    Includes insights parsed implicitly from prior EDA analyses.
    """
    if ts_mandal is None or ts_mandal.empty:
        return []

    forecast_results = []
    
    # Read EDA results to guide model selection dynamically
    overdispersed_diseases = []
    try:
        eda_stats = pd.read_csv("eda_output/time_series_stats.csv")
        for _, row in eda_stats.iterrows():
            if "overdispersed" in str(row.get("distribution_hint", "")):
                overdispersed_diseases.append(row["disease_key"])
    except:
        pass
    
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

        # --- NegBin GLM Pipeline dynamically invoked for Overdispersed profiles ---
        if d_key in overdispersed_diseases and len(train) > 10:
            try:
                # Basic NegBin logic wrapper
                df_nb = pd.DataFrame({'case_count': y})
                df_nb['lag_1'] = df_nb['case_count'].shift(1)
                df_nb['lag_2'] = df_nb['case_count'].shift(2)
                df_nb = df_nb.dropna()
                if not df_nb.empty:
                    X_train_nb = sm.add_constant(df_nb[['lag_1', 'lag_2']].iloc[:-forecast_horizon])
                    Y_train_nb = df_nb['case_count'].iloc[:-forecast_horizon]
                    if len(X_train_nb) > 5:
                        model_nb = sm.GLM(Y_train_nb, X_train_nb, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
                        
                        # Test forecast
                        X_test_nb = sm.add_constant(df_nb[['lag_1', 'lag_2']].iloc[-forecast_horizon:], has_constant='add')
                        preds_nb = model_nb.predict(X_test_nb)
                        models_performance['NegBin GLM'] = np.sqrt(mean_squared_error(test[-len(preds_nb):], preds_nb))
                        
                        # Future Forecast Needs basic recursive stepping but for simplicity we rely recursively 
                        # just like XGBoost
                        final_nb_preds = []
                        last_lags = y[-2:].tolist()
                        for _ in range(forecast_horizon):
                            pred_val = model_nb.predict([1, last_lags[1], last_lags[0]])[0]
                            final_nb_preds.append(pred_val)
                            last_lags.append(pred_val)
                            last_lags.pop(0)
                        predictions_future['NegBin GLM'] = final_nb_preds
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
    if df.empty:
        latest_cutoff = pd.Timestamp.now()
        recent = df
    else:
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
            
    # Generate Alerts via Rule Engine
    try:
        from rule_engine import evaluate_rules
        ts_weekly = df.groupby([pd.Grouper(key='event_date', freq='W-MON'), 'disease_key']).size().reset_index(name='case_count')
        ts_weekly.rename(columns={'event_date': 'period'}, inplace=True)
        alerts_df = evaluate_rules(df, ts_weekly, ts)
        
        alerts_by_tier = {"tier1": [], "tier2": [], "tier3": []}
        for _, row in alerts_df.iterrows():
            level = str(row['level'])
            tier_key = "tier1" if level == "P0" else ("tier2" if level == "P1" else "tier3")
            alerts_by_tier[tier_key].append({
                "id": str(_),
                "disease": row['disease'],
                "mandal": row['mandal'],
                "cases": int(row['cases']),
                "startDate": row['onset_date'].strftime("%Y-%m-%d") if pd.notnull(row['onset_date']) else "",
                "severity": row['rule_name'],
                "facilities": row.get('facilities', '')
            })
            
        with open('frontend/public/alerts.json', 'w') as f:
            json.dump(alerts_by_tier, f, indent=2)
    except Exception as e:
        print(f"Failed generating alerts: {e}")
            
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
    import sys
    print("Phase 1: Loading raw schema and merging multi-year sets...")
    # The default can look for multi-year files recursively
    df = load_and_preprocess(["2024.csv", "2025.csv"])
    
    if not df.empty:
        print("Phase 1.5: Executing EDA Runner to identify model characteristics (Overdispersion, Missing Vitals)...")
        try:
            from eda_runner import run_full_eda
            run_full_eda(df)
        except Exception as e:
            print(f"Skipping complete EDA output generation due to dependencies: {e}")
            
        print("Phase 2: Aggregating temporal mandal streams...")
        ts_mandal = get_mandal_timeseries(df)
        
        print("Phase 3: Deploying Auto-ARIMA & other models natively...")
        forecasts = run_mandal_forecasts(ts_mandal, forecast_horizon=4)
        
        print("Phase 4: Converting formats and dumping JSONs...")
        generate_frontend_assets(df, ts_mandal, forecasts)
    else:
        print("Terminating pipeline: Loaded dataframe is completely empty.")
