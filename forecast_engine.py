"""
Forecast Engine — Production Forecasting Pipeline
===================================================
Runs the dynamic CV-winning model per disease to produce 4-week forecasts
with 5th/50th/95th percentile scenario bands. Also performs Mandal-level
forecasting for localized early warnings.
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    import statsmodels.api as sm
except ImportError:
    pass

try:
    from prophet import Prophet
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
except ImportError:
    pass

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    pass

from config import DISEASE_CODES

# =========================================================================
# STATIC MODEL REGISTRY (Fallback)
# =========================================================================
STATIC_MODEL_REGISTRY = {
    "mud_fever":       {"model": "arima",    "order": (1, 1, 1)},
    "gastroenteritis": {"model": "ucm",      "covariates": ["mandal_count"]},
    "dengue":          {"model": "negbin",    "covariates": ["mandal_count", "spo2"]},
    "malaria":         {"model": "hw",        "seasonal": False},
    "chikungunya":     {"model": "hw",        "seasonal": False},
    "cholera":         {"model": "arima",     "order": (1, 1, 1)},
    "typhoid":         {"model": "ucm",       "covariates": ["mandal_count"]},
    "febrile_illness": {"model": "arima",     "order": (1, 1, 1)},
}

def load_dynamic_config():
    """Load latest winning models from CV results if available."""
    dynamic_registry = STATIC_MODEL_REGISTRY.copy()
    cv_path = "model_output/cv_results_summary.csv"
    if os.path.exists(cv_path):
        try:
            cv_df = pd.read_csv(cv_path)
            best_idx = cv_df.groupby("disease")["rmse"].idxmin()
            best_models = cv_df.loc[best_idx]
            
            def _map_model_name(name):
                if "ARIMA" in name or "SARIMA" in name: return {"model": "arima", "order": (1,1,1)}
                if "NegBin" in name: return {"model": "negbin", "covariates": ["mandal_count", "spo2"]}
                if "XGBoost" in name: return {"model": "xgboost"}
                if "Random Forest" in name: return {"model": "rf"}
                if "Prophet" in name: return {"model": "prophet"}
                if "UCM" in name or "BSTS" in name: return {"model": "ucm", "covariates": ["mandal_count"]}
                if "Holt" in name: return {"model": "hw", "seasonal": False}
                return None

            name_to_key = {info["name"]: key for key, info in DISEASE_CODES.items()}
            
            for _, row in best_models.iterrows():
                disease_name = row["disease"]
                model_str = row["model"]
                key = name_to_key.get(disease_name)
                
                config = _map_model_name(model_str)
                if key and config:
                    dynamic_registry[key] = config
            print("  [Forecast] Loaded dynamic configuration from CV results.")
        except Exception as e:
            print(f"  [Forecast] Error loading CV results, using static. {e}")
    return dynamic_registry

MODEL_REGISTRY = load_dynamic_config()


def forecast_disease(disease_ts, disease_key, horizon=4):
    """Generate a forecast for a given time series slice."""
    if disease_key not in MODEL_REGISTRY:
        return None

    config = MODEL_REGISTRY[disease_key]
    disease_ts = disease_ts.sort_values("period").reset_index(drop=True)

    if disease_ts.empty or len(disease_ts) < 5:
        return None

    disease_name = disease_ts["disease_name"].iloc[0]
    series = pd.Series(
        disease_ts["case_count"].values.astype(float),
        index=pd.DatetimeIndex(disease_ts["period"]),
    )

    last_date = series.index[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")

    model_type = config["model"]
    result = {
        "disease_key": disease_key,
        "disease_name": disease_name,
        "model_name": "",
        "forecast_dates": forecast_dates,
        "predicted": None,
        "lower_95": None,
        "upper_95": None,
        "lower_50": None,
        "upper_50": None,
        "historical_dates": series.index,
        "historical_values": series.values,
    }

    try:
        if model_type == "arima":
            result = _forecast_arima(series, horizon, config, result)
        elif model_type == "ucm":
            result = _forecast_ucm(series, disease_ts, horizon, config, result)
        elif model_type == "negbin":
            result = _forecast_negbin(disease_ts, horizon, config, result)
        elif model_type == "hw":
            result = _forecast_hw(series, horizon, config, result)
        elif model_type == "xgboost":
            result = _forecast_xgboost_rf(disease_ts, horizon, config, result, use_rf=False)
        elif model_type == "rf":
            result = _forecast_xgboost_rf(disease_ts, horizon, config, result, use_rf=True)
        elif model_type == "prophet":
            result = _forecast_prophet(disease_ts, horizon, config, result)
    except Exception as e:
        # Fallback to naive moving average
        avg = np.mean(series.tail(4))
        std = np.std(series.tail(12)) if len(series) > 12 else np.std(series)
        if np.isnan(std) or std == 0: std = 1.0
        
        result["model_name"] = "Naive (fallback)"
        result["predicted"] = np.full(horizon, avg)
        result["lower_95"] = np.maximum(np.full(horizon, avg - 1.96 * std), 0)
        result["upper_95"] = np.full(horizon, avg + 1.96 * std)
        result["lower_50"] = np.maximum(np.full(horizon, avg - 0.67 * std), 0)
        result["upper_50"] = np.full(horizon, avg + 0.67 * std)

    for key in ["predicted", "lower_95", "upper_95", "lower_50", "upper_50"]:
        if result[key] is not None:
            arr = np.asarray(result[key], dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            result[key] = np.maximum(arr, 0)

    return result

# ---------------------------------------------------------
# MODEL IMPLEMENTATIONS
# ---------------------------------------------------------
def _forecast_arima(series, horizon, config, result):
    order = config.get("order", (1, 1, 1))
    model = SARIMAX(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=200)
    forecast = fit.get_forecast(horizon)
    result["model_name"] = f"ARIMA{order}"
    result["predicted"] = np.asarray(forecast.predicted_mean, dtype=float)
    ci_95 = forecast.conf_int(alpha=0.05)
    ci_50 = forecast.conf_int(alpha=0.50)
    result["lower_95"] = np.asarray(ci_95.iloc[:, 0], dtype=float)
    result["upper_95"] = np.asarray(ci_95.iloc[:, 1], dtype=float)
    result["lower_50"] = np.asarray(ci_50.iloc[:, 0], dtype=float)
    result["upper_50"] = np.asarray(ci_50.iloc[:, 1], dtype=float)
    return result


def _forecast_ucm(series, disease_ts, horizon, config, result):
    cov_cols = [c for c in config.get("covariates", []) if c in disease_ts.columns]
    spec = {"level": "local linear trend"}
    if len(series) >= 104: spec["seasonal"] = 52
    
    if cov_cols:
        cov_data = disease_ts[cov_cols].fillna(method="ffill").fillna(0).values.astype(float)
        spec["exog"] = cov_data
        result["model_name"] = "UCM+covariates"
    else:
        result["model_name"] = "UCM"

    model = UnobservedComponents(series, **spec)
    fit = model.fit(disp=False, maxiter=300)

    forecast_kwargs = {}
    if cov_cols:
        forecast_kwargs["exog"] = cov_data[-1:].repeat(horizon, axis=0)

    forecast = fit.get_forecast(horizon, **forecast_kwargs)
    result["predicted"] = np.asarray(forecast.predicted_mean, dtype=float)
    ci_95 = forecast.conf_int(alpha=0.05)
    ci_50 = forecast.conf_int(alpha=0.50)
    result["lower_95"] = np.asarray(ci_95.iloc[:, 0], dtype=float)
    result["upper_95"] = np.asarray(ci_95.iloc[:, 1], dtype=float)
    result["lower_50"] = np.asarray(ci_50.iloc[:, 0], dtype=float)
    result["upper_50"] = np.asarray(ci_50.iloc[:, 1], dtype=float)
    return result


def _forecast_negbin(disease_ts, horizon, config, result):
    df = disease_ts.copy()
    df["lag_1"] = df["case_count"].shift(1)
    df["lag_2"] = df["case_count"].shift(2)
    df["lag_4"] = df["case_count"].shift(4)
    df["rolling_4"] = df["case_count"].shift(1).rolling(4, min_periods=1).mean()
    df["month"] = pd.to_datetime(df["period"]).dt.month

    for m in range(2, 13): df[f"m_{m}"] = (df["month"] == m).astype(float)
    month_cols = [f"m_{m}" for m in range(2, 13)]
    feature_cols = ["lag_1", "lag_2", "lag_4", "rolling_4"] + month_cols

    cov_cols = [c for c in config.get("covariates", []) if c in df.columns]
    for cov in cov_cols:
        if df[cov].notna().sum() > len(df) * 0.3:
            df[cov] = df[cov].fillna(df[cov].median())
            feature_cols.append(cov)

    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "case_count"])
    if len(df) < 5: raise ValueError("Not enough lag data")

    X = np.column_stack([np.ones(len(df)), df[feature_cols].values.astype(float)])
    y = df["case_count"].values.astype(float)

    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0))
    fit = model.fit()
    resid_std = np.nanstd(np.asarray(fit.resid_response, dtype=float))

    predictions = []
    last_known = list(df["case_count"].values[-4:])
    last_date = pd.to_datetime(df["period"].max())
    
    for step in range(horizon):
        future_date = last_date + pd.Timedelta(weeks=step + 1)
        future_month = future_date.month
        lag_1 = last_known[-1]
        lag_2 = last_known[-2] if len(last_known) >= 2 else lag_1
        lag_4 = last_known[-4] if len(last_known) >= 4 else lag_1
        rolling_4 = np.mean(last_known[-4:])
        feat_vals = [lag_1, lag_2, lag_4, rolling_4] + [(1.0 if future_month == m else 0.0) for m in range(2, 13)]
        for cov in cov_cols:
            if cov in feature_cols: feat_vals.append(float(df[cov].iloc[-1]))
        x_new = np.array([1.0] + feat_vals).reshape(1, -1)
        pred_val = float(fit.predict(x_new)[0])
        predictions.append(max(pred_val, 0))
        last_known.append(pred_val)

    pred = np.array(predictions)
    result["model_name"] = "NegBin GLM"
    result["predicted"] = pred
    result["lower_95"] = np.maximum(pred - 1.96 * resid_std, 0)
    result["upper_95"] = pred + 1.96 * resid_std
    result["lower_50"] = np.maximum(pred - 0.67 * resid_std, 0)
    result["upper_50"] = pred + 0.67 * resid_std
    return result


def _forecast_hw(series, horizon, config, result):
    seasonal = config.get("seasonal", False)
    if seasonal and len(series) >= 104:
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=52, initialization_method="estimated")
        result["model_name"] = "Holt-Winters"
    else:
        model = ExponentialSmoothing(series, trend="add", damped_trend=True, initialization_method="estimated")
        result["model_name"] = "Holt-Winters (damped)"

    fit = model.fit(optimized=True, use_brute=True)
    pred = np.asarray(fit.forecast(horizon), dtype=float)
    resid_std = np.nanstd(np.asarray(fit.resid, dtype=float))

    result["predicted"] = np.maximum(pred, 0)
    result["lower_95"] = np.maximum(pred - 1.96 * resid_std, 0)
    result["upper_95"] = pred + 1.96 * resid_std
    result["lower_50"] = np.maximum(pred - 0.67 * resid_std, 0)
    result["upper_50"] = pred + 0.67 * resid_std
    return result


def _forecast_xgboost_rf(disease_ts, horizon, config, result, use_rf=False):
    df = disease_ts.copy()
    for lag in [1, 2, 4]: df[f"lag_{lag}"] = df["case_count"].shift(lag)
    df["rolling_4"] = df["case_count"].shift(1).rolling(4, min_periods=1).mean()
    df["month"] = pd.to_datetime(df["period"]).dt.month
    
    feature_cols = ["lag_1", "lag_2", "lag_4", "rolling_4", "month"]
    covariates = config.get("covariates", [])
    cov_cols = [c for c in covariates if c in df.columns]
    feature_cols.extend(cov_cols)
    
    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "case_count"])
    if len(df) < 5: raise ValueError("Not enough training rows")
    
    X = df[feature_cols].values
    y = df["case_count"].values
    
    if use_rf:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        result["model_name"] = "Random Forest"
    else:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        result["model_name"] = "XGBoost"
        
    model.fit(X, y)
    preds_train = model.predict(X)
    resid_std = np.nanstd(y - preds_train)
    
    predictions = []
    last_known = list(disease_ts["case_count"].values[-4:])
    last_date = pd.to_datetime(disease_ts["period"].max())
    
    for step in range(horizon):
        future_date = last_date + pd.Timedelta(weeks=step + 1)
        lag_1 = last_known[-1]
        lag_2 = last_known[-2] if len(last_known) >= 2 else lag_1
        lag_4 = last_known[-4] if len(last_known) >= 4 else lag_1
        rolling_4 = np.mean(last_known[-4:])
        feat_vals = [lag_1, lag_2, lag_4, rolling_4, future_date.month]
        for cov in cov_cols: feat_vals.append(float(df[cov].iloc[-1]))
            
        x_new = np.array([feat_vals])
        pred_val = float(model.predict(x_new)[0])
        predictions.append(max(pred_val, 0))
        last_known.append(pred_val)
        
    pred = np.array(predictions)
    result["predicted"] = pred
    result["lower_95"] = np.maximum(pred - 1.96 * resid_std, 0)
    result["upper_95"] = pred + 1.96 * resid_std
    result["lower_50"] = np.maximum(pred - 0.67 * resid_std, 0)
    result["upper_50"] = pred + 0.67 * resid_std
    return result


def _forecast_prophet(disease_ts, horizon, config, result):
    df_p = disease_ts[["period", "case_count"]].copy().rename(columns={"period": "ds", "case_count": "y"})
    covariates = config.get("covariates", [])
    cov_cols = [c for c in covariates if c in disease_ts.columns]
    for cov in cov_cols: df_p[cov] = disease_ts[cov]
        
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for cov in cov_cols: m.add_regressor(cov)
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=horizon, freq='W')
    for cov in cov_cols:
        last_cov = disease_ts[cov].iloc[-1]
        future[cov] = future["ds"].apply(lambda d: disease_ts.loc[disease_ts["period"]==d, cov].iloc[0] if d <= disease_ts["period"].max() else last_cov)
        
    fcst = m.predict(future)
    pred_future = fcst.tail(horizon)
    
    pred = np.maximum(pred_future["yhat"].values, 0)
    result["model_name"] = "Prophet"
    result["predicted"] = pred
    result["lower_95"] = np.maximum(pred_future["yhat_lower"].values, 0)
    result["upper_95"] = pred_future["yhat_upper"].values
    # Approximate 50% CI by scaling 95% width
    width = result["upper_95"] - result["lower_95"]
    result["lower_50"] = np.maximum(pred - width*0.25, 0)
    result["upper_50"] = pred + width*0.25
    return result


# =========================================================================
# BATCH FORECAST
# =========================================================================
def forecast_all(ts_weekly, ts_mandal_weekly=None, horizon=4):
    """
    Run forecasts for state-level and optionally mandal-level.
    """
    state_results = {}
    mandal_results = []
    
    # Core state-level forecasts
    for disease_key in MODEL_REGISTRY:
        print(f"  Forecasting {disease_key} (State)...", end=" ")
        disease_ts = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
        if disease_ts.empty: 
            print("✗ No data")
            continue
            
        res = forecast_disease(disease_ts, disease_key, horizon)
        if res and res["predicted"] is not None:
            state_results[disease_key] = res
            print(f"✓ {res['model_name']} — "
                  f"predicted next {horizon}w: {[round(x, 1) for x in res['predicted']]}")
        else:
            print("✗ Failed")
            
    # Mandal-level forecasts
    if ts_mandal_weekly is not None and not ts_mandal_weekly.empty:
        print("  Generating Mandal-level forecasts...")
        for disease_key in MODEL_REGISTRY:
            disease_ts_m = ts_mandal_weekly[ts_mandal_weekly["disease_key"] == disease_key].copy()
            if disease_ts_m.empty: continue
            
            # Filter mandals by minimum historical impact (at least 5 weeks recorded vs total silence)
            mandal_counts = disease_ts_m["mandal"].value_counts()
            active_mandals = mandal_counts[mandal_counts >= 5].index
            
            for mandal in active_mandals:
                m_ts = disease_ts_m[disease_ts_m["mandal"] == mandal].copy()
                if m_ts.empty or m_ts["case_count"].sum() < 3: 
                    continue # Skip practically empty mandals
                
                # Fetch dynamically selected default config for this disease
                m_res = forecast_disease(m_ts, disease_key, horizon)
                if m_res and m_res["predicted"] is not None:
                    # Append rows directly to a master list
                    for i, date in enumerate(m_res["forecast_dates"]):
                        mandal_results.append({
                            "period": date,
                            "disease_key": disease_key,
                            "disease_name": m_res["disease_name"],
                            "mandal": mandal,
                            "district": m_ts.iloc[-1]["district"] if "district" in m_ts.columns else "Unknown",
                            "case_count": m_res["predicted"][i]
                        })

    return state_results, mandal_results
