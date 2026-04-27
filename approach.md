# Disease Outbreak Forecasting Approach

This document outlines the methodological approach to the forecasting engine, specifically detailing the features (attributes) engineered and the diverse suite of models evaluated and utilized for predictive analytics.

## 1. Attributes Used (Features & Covariates)

The predictive engines run on aggregated weekly time-series data. Features are grouped into endogenous time-series transformations and exogenous clinical/locational covariates. 

### 1.1 Core Temporal & Auto-Regressive Features
To enable machine learning and GLM models to understand temporal dynamics, the system generates lagging and smoothing features:
- **`case_count`**: The core dependent variable targeted for forecasting.
- **`lag_1`, `lag_2`, `lag_4`**: Lagged case counts (from 1, 2, and 4 weeks prior) to capture short-term and medium-term autocorrelation.
- **`rolling_4`**: A 4-week rolling average for smoothing out short-term volatility and capturing immediate prior trends.
- **`month`**: Month of the year extracted from the period, heavily utilized (and one-hot encoded as `m_2` through `m_12` in GLMs) to capture implicit annual seasonality constraints.

### 1.2 Clinical & Locational Covariates
Specific exogenous variables are integrated into multivariate models depending on the epidemiological characteristics of the disease (as identified during Exploratory Data Analysis):
- **`mandal_count`**: The number of unique mandals reporting cases in a given week. This acts as a proxy for the geographic spread and velocity of an outbreak. ( Heavily used for Gastroenteritis, Mud Fever, Dengue, etc.)
- **`spo2`**: Aggregate blood oxygen levels. Found to be relevant for particular viral/vector-borne severe outbreaks (e.g., Dengue).
- **`duration_days`**: Average duration of patient symptoms prior to clinic presentation. Helpful in tracking the maturation of an outbreak cluster (e.g., Mud Fever).

*Note: The raw pipeline processes various other detailed clinical attributes (vitals like `temperature`, `pulse`, `systole`; demographics like `district`, `mandal`; and patient symptoms like `severity`). However, the forecasting engine relies primarily on the high-level aggregated covariates above to combat overfitting and data sparsity constraints prior to prediction.*

---

## 2. Models Used

No single forecasting algorithm performs best uniformly across disparate diseases with varying properties (e.g., some are trend-stationary, others difference-stationary or overdispersed). The system addresses this by leveraging a heterogeneous ensemble of statistical, GLM, and Machine Learning models.

### 2.1 Statistical & State-Space Models
- **Holt-Winters Exponential Smoothing**: Decomposes the time series into level, trend, and distinct seasonal (52-week) components. Suitable for diseases with clear cyclical patterns and sufficient historical depth.
- **SARIMA (Seasonal Auto-Regressive Integrated Moving Average)**: A classical framework applied for baseline univariate forecasting, modeling partial autocorrelations and moving averages.
- **UCM (Unobserved Components Model / BSTS-like)**: Models the series via a local linear trend and structural seasonal factors while gracefully accepting external covariates (`mandal_count`). Highly resilient to structural breaks in reporting data.

### 2.2 Generalized Linear Models
- **Negative Binomial GLM**: Count data for outbreaks heavily violates normality due to overdispersion (variance > mean). The Negative Binomial GLM is constructed using lag features and covariates directly, making it exceptionally reliable for diseases exhibiting sparse baseline behavior but sudden, violent surges.

### 2.3 Machine Learning Regressors
- **XGBoost**: Gradient Boosted Decision Trees trained over structured feature spaces (`lag_1`, `lag_2`, `lag_4`, `rolling_4`, `month`, and covariates). Excellent at mapping non-linear interactions without assuming prior distributions.
- **Random Forest**: An ensemble bagging algorithm utilized for generating robust, low-variance predictions with the same tabular feature space as XGBoost.

### 2.4 Hybrid / Proprietary Frameworks
- **Prophet**: Meta's forecasting framework based on an additive model where non-linear trends are fit with yearly and weekly seasonality. It natively incorporates the exogenous covariates as regressors.

---

## 3. Dynamic Model Selection Framework

To ensure the forecast remains highly accurate as disease characteristics evolve over time, the system avoids hard-coding a single model. 

1. **Walk-Forward Cross-Validation**: During batch training, the system slices the historical timeline using walk-forward validation (respecting the temporal arrow). 
2. **Evaluation Metrics**: Each candidate algorithm is trained on past data and evaluated on hold-out horizons. Models are scored on **RMSE** (Root Mean Square Error), **MAE** (Mean Absolute Error), and **Coverage** of 95% Confidence Intervals.
3. **Dynamic Registry**: The model that achieves the best out-of-sample RMSE for a specific disease is dynamically selected and saved to `cv_results_summary.csv`.
4. **Prediction Context**: Production inferences are made using these winning algorithms to emit P50 and P95 percentile bounds, establishing worst-case/best-case scenario bands for administrators at both state levels and localized mandal levels. A static algorithmic fallback is provided should historical data be temporarily insufficient.
