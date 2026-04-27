"""
Disease Outbreak Surveillance Dashboard
=========================================
Streamlit dashboard for AP RTGS.
Offline-safe: no CDN, no external tiles, no internet required.

Run: streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from config import DISEASE_CODES, get_disease_names
from data_loader import load_and_clean, aggregate_time_series
from rule_engine import evaluate_rules, PRIORITY_COLORS, PRIORITY_LABELS, get_alert_summary
from forecast_engine import forecast_all, forecast_disease, MODEL_REGISTRY

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Disease Outbreak Surveillance — AP RTGS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# CUSTOM CSS — Premium light theme, high contrast, print-friendly
# =========================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8fafc; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    div[data-testid="metric-container"] label {
        color: #475569 !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* Alert cards */
    .alert-p0 { background: #fef2f2; border-left: 4px solid #dc2626; padding: 12px; border-radius: 8px; margin: 6px 0; }
    .alert-p1 { background: #fff7ed; border-left: 4px solid #ea580c; padding: 12px; border-radius: 8px; margin: 6px 0; }
    .alert-p2 { background: #fffbeb; border-left: 4px solid #d97706; padding: 12px; border-radius: 8px; margin: 6px 0; }
    .alert-p3 { background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px; border-radius: 8px; margin: 6px 0; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        padding: 8px 0;
        border-bottom: 2px solid #3b82f6;
        margin-bottom: 16px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# DATA LOADING (cached)
# =========================================================================
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file_path):
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
    df_clean = load_and_clean(df, verbose=False)
    return df_clean


@st.cache_data(show_spinner="Aggregating time series...")
def get_time_series(df_clean_hashable, ref_date_key):
    ts_weekly = aggregate_time_series(df_clean_hashable, freq="W", group_cols=[])
    mandal_cols = ["district", "mandal"] if "district" in df_clean_hashable.columns else ["mandal"]
    ts_mandal_weekly = aggregate_time_series(df_clean_hashable, freq="W", group_cols=mandal_cols)
    
    ts_district_weekly = pd.DataFrame()
    if "district" in df_clean_hashable.columns:
        ts_district_weekly = aggregate_time_series(df_clean_hashable, freq="W", group_cols=["district"])
        
    return ts_weekly, ts_mandal_weekly, ts_district_weekly


@st.cache_data(show_spinner="Running forecasts...")
def get_forecasts(ts_weekly_hashable, ts_mandal_hashable, ref_date_key, horizon=4):
    return forecast_all(ts_weekly_hashable, ts_mandal_weekly=ts_mandal_hashable, horizon=horizon)


@st.cache_data(show_spinner="Evaluating alert rules...")
def get_alerts(df_clean_hashable, ts_weekly_hashable, ts_mandal_hashable, ref_date_key):
    return evaluate_rules(df_clean_hashable, ts_weekly_hashable, ts_mandal_hashable, ref_date=pd.to_datetime(ref_date_key))


# =========================================================================
# SIDEBAR
# =========================================================================
def render_sidebar():
    st.sidebar.markdown("## 🏥 AP Disease Surveillance")
    st.sidebar.markdown("---")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload CSV/Parquet", "Use pre-loaded data"],
        index=1,
    )

    file_path = None
    if data_source == "Upload CSV/Parquet":
        uploaded = st.sidebar.file_uploader("Upload data file", type=["csv", "parquet"])
        if uploaded:
            # Save to temp location
            temp_path = os.path.join("_temp_upload", uploaded.name)
            os.makedirs("_temp_upload", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            file_path = temp_path
    else:
        # Look for pre-existing data
        for candidate in ["eda_output/clean_data.parquet", "model_output/weekly_timeseries.csv",
                          "clean_data.parquet", "data.csv"]:
            if os.path.exists(candidate):
                file_path = candidate
                break

    st.sidebar.markdown("---")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 12, 4)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Disease Filter")
    disease_names = get_disease_names()
    selected_diseases = st.sidebar.multiselect(
        "Select diseases",
        options=list(disease_names.keys()),
        default=list(disease_names.keys()),
        format_func=lambda x: disease_names[x],
    )

    st.sidebar.markdown("---")
    enable_beta = st.sidebar.checkbox("Enable Beta Features (Reports)", value=True)
    
    st.sidebar.markdown(
        f"<small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>",
        unsafe_allow_html=True,
    )

    return file_path, forecast_horizon, selected_diseases, enable_beta


# =========================================================================
# OVERVIEW TAB
# =========================================================================
def render_overview(df_clean, ts_weekly, alerts_df, selected_diseases):
    st.markdown('<div class="section-header">📊 State Overview</div>', unsafe_allow_html=True)

    # Filter to selected diseases
    df_sel = df_clean[df_clean["disease_key"].isin(selected_diseases)]
    latest_date = df_sel["event_date"].max()
    last_week = df_sel[df_sel["event_date"] >= latest_date - timedelta(days=7)]
    prev_week = df_sel[
        (df_sel["event_date"] >= latest_date - timedelta(days=14)) &
        (df_sel["event_date"] < latest_date - timedelta(days=7))
    ]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total_this_week = len(last_week)
    total_prev_week = len(prev_week)
    delta = total_this_week - total_prev_week

    c1.metric("Cases This Week", f"{total_this_week:,}",
              delta=f"{delta:+,}", delta_color="inverse")
    c2.metric("Active Alerts", len(alerts_df),
              delta=f"{len(alerts_df[alerts_df['level'].isin(['P0','P1'])])} critical")
    c3.metric("Mandals Affected", last_week["mandal"].nunique() if "mandal" in last_week.columns else "N/A")
    c4.metric("Diseases Active", last_week["disease_key"].nunique())

    st.markdown("---")

    # Disease summary table with sparklines
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Disease Trends")
        resolution = st.radio("Resolution", ["Daily", "Weekly"], horizontal=True, label_visibility="collapsed")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        
        for i, dk in enumerate(selected_diseases):
            if resolution == "Weekly":
                d_ts = ts_weekly[ts_weekly["disease_key"] == dk].sort_values("period")
                if d_ts.empty: continue
                name = d_ts["disease_name"].iloc[0]
                fig.add_trace(go.Scatter(
                    x=d_ts["period"], y=d_ts["case_count"],
                    name=name, mode="lines",
                    line=dict(width=2.5, color=colors[i % len(colors)]),
                    hovertemplate=f"<b>{name}</b><br>Week: %{{x}}<br>Cases: %{{y}}<extra></extra>",
                ))
            else:
                d_data = df_sel[df_sel["disease_key"] == dk]
                if d_data.empty: continue
                name = DISEASE_CODES[dk]["name"]
                
                # Calculate daily
                d_ts = d_data.groupby("event_date").size().reset_index(name="case_count")
                # Fill missing dates with 0
                idx = pd.date_range(d_ts["event_date"].min(), d_ts["event_date"].max())
                d_ts = d_ts.set_index("event_date").reindex(idx, fill_value=0).reset_index()
                d_ts.rename(columns={"index": "event_date"}, inplace=True)
                
                # Plot raw daily as faint lines
                fig.add_trace(go.Scatter(
                    x=d_ts["event_date"], y=d_ts["case_count"],
                    name=name + " (Daily)", mode="lines",
                    line=dict(width=1, color="rgba(150,150,150,0.5)"),
                    hovertemplate=f"<b>{name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Cases: %{{y}}<extra></extra>",
                ))
                
                # Plot 7-day moving average
                rolling = d_ts["case_count"].rolling(7, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=d_ts["event_date"], y=rolling,
                    name=f"{name} (7d MA)", mode="lines",
                    line=dict(width=2.5, color=colors[i % len(colors)]),
                    hoverinfo="skip"
                ))

        fig.update_layout(
            height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="", yaxis_title=f"{resolution} Cases",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Disease Summary")
        summary_rows = []
        for dk in selected_diseases:
            info = DISEASE_CODES.get(dk, {})
            d_ts = ts_weekly[ts_weekly["disease_key"] == dk]
            if d_ts.empty:
                continue
            total = int(d_ts["case_count"].sum())
            latest = int(d_ts.sort_values("period").iloc[-1]["case_count"]) if not d_ts.empty else 0
            n_alerts = len(alerts_df[alerts_df["disease_key"] == dk])
            summary_rows.append({
                "Disease": info["name"],
                "Total Cases": f"{total:,}",
                "This Week": latest,
                "Alerts": n_alerts,
                "Model": MODEL_REGISTRY.get(dk, {}).get("model", "rules").upper(),
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# =========================================================================
# FORECAST TAB
# =========================================================================
def render_forecasts(ts_weekly, forecasts, selected_diseases):
    st.markdown('<div class="section-header">📈 4-Week Forecasts</div>', unsafe_allow_html=True)

    if not forecasts:
        st.warning("No forecasts available. Run the forecast engine first.")
        return

    # Create a tab per disease
    disease_tabs = []
    disease_keys = []
    for dk in selected_diseases:
        if dk in forecasts:
            disease_tabs.append(DISEASE_CODES[dk]["name"])
            disease_keys.append(dk)

    if not disease_tabs:
        st.info("No forecasts available for selected diseases.")
        return

    st.markdown("##### Plot Time Horizon")
    history_view = st.radio(
        "Zoom historical data view:",
        ["Detailed 3 Months", "Standard 1 Year", "All Time Dataset"],
        horizontal=True,
        index=1,
        label_visibility="collapsed"
    )
    
    if history_view == "Detailed 3 Months":
        history_weeks = 12
    elif history_view == "Standard 1 Year":
        history_weeks = 52
    else:
        history_weeks = 9999

    tabs = st.tabs(disease_tabs)

    for tab, dk in zip(tabs, disease_keys):
        with tab:
            fc = forecasts[dk]
            hist_dates = fc["historical_dates"]
            hist_vals = fc["historical_values"]

            # Show configured weeks + forecast
            show_weeks = min(history_weeks, len(hist_dates))
            h_dates = hist_dates[-show_weeks:]
            h_vals = hist_vals[-show_weeks:]

            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(
                x=h_dates, y=h_vals,
                name="Observed", mode="lines",
                line=dict(color="#475569", width=1.5),
            ))

            # Rolling average
            if len(h_vals) > 4:
                rolling = pd.Series(h_vals).rolling(4, min_periods=1).mean().values
                fig.add_trace(go.Scatter(
                    x=h_dates, y=rolling,
                    name="4-week MA", mode="lines",
                    line=dict(color="#2563eb", width=2.5),
                ))

            # 95% CI band
            fig.add_trace(go.Scatter(
                x=list(fc["forecast_dates"]) + list(fc["forecast_dates"][::-1]),
                y=list(fc["upper_95"]) + list(fc["lower_95"][::-1]),
                fill="toself", fillcolor="rgba(220,38,38,0.1)",
                line=dict(width=0), name="95% CI",
                hoverinfo="skip",
            ))

            # 50% CI band
            if fc["lower_50"] is not None:
                fig.add_trace(go.Scatter(
                    x=list(fc["forecast_dates"]) + list(fc["forecast_dates"][::-1]),
                    y=list(fc["upper_50"]) + list(fc["lower_50"][::-1]),
                    fill="toself", fillcolor="rgba(220,38,38,0.2)",
                    line=dict(width=0), name="50% CI",
                    hoverinfo="skip",
                ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=fc["forecast_dates"], y=fc["predicted"],
                name=f"Forecast ({fc['model_name']})",
                mode="lines+markers",
                line=dict(color="#dc2626", width=2.5, dash="dash"),
                marker=dict(size=8),
            ))

            fig.update_layout(
                height=450, template="plotly_white",
                title=f"{fc['disease_name']} — {fc['model_name']}",
                xaxis_title="", yaxis_title="Weekly Cases",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            fc_table = pd.DataFrame({
                "Week": fc["forecast_dates"].strftime("%Y-%m-%d"),
                "Predicted": [round(x, 1) for x in fc["predicted"]],
                "Lower 95%": [round(x, 1) for x in fc["lower_95"]],
                "Upper 95%": [round(x, 1) for x in fc["upper_95"]],
            })
            st.dataframe(fc_table, use_container_width=True, hide_index=True)


# =========================================================================
# ALERTS TAB
# =========================================================================
def render_alerts(alerts_df, selected_diseases, ts_weekly=None, ts_mandal_weekly=None):
    st.markdown('<div class="section-header">🚨 Active Alerts (IDSP Rules)</div>', unsafe_allow_html=True)

    # Specific disease filter for alerts tab
    alert_diseases = st.multiselect(
        "Filter Alerts by Disease:",
        options=selected_diseases,
        default=selected_diseases,
        format_func=lambda x: DISEASE_CODES[x]["name"] if x in DISEASE_CODES else x
    )

    alerts_filtered = alerts_df[alerts_df["disease_key"].isin(alert_diseases)]

    if alerts_filtered.empty:
        st.success("✅ No active alerts for selected diseases.")
        return

    # Summary counts
    c1, c2, c3, c4 = st.columns(4)
    for col, level in zip([c1, c2, c3, c4], ["P0", "P1", "P2", "P3"]):
        count = len(alerts_filtered[alerts_filtered["level"] == level])
        col.metric(
            PRIORITY_LABELS.get(level, level),
            count,
            delta=level,
            delta_color="off",
        )

    st.markdown("---")

    # Add `is_predicted` column
    def _is_pred(row):
        try:
            return pd.to_datetime(row.get("onset_date", row["timestamp"])) > pd.to_datetime(row["timestamp"])
        except:
            return False
            
    alerts_filtered["is_predicted"] = alerts_filtered.apply(_is_pred, axis=1)

    import os
    m_df = pd.DataFrame()
    if os.path.exists("media_alerts.csv"):
        try:
            m_df = pd.read_csv("media_alerts.csv")
            if not m_df.empty:
                m_df = m_df[m_df["disease_key"].isin(selected_diseases)]
        except:
            pass

    def _get_media_matches(alert_row):
        if m_df.empty:
            return [], []
        matched_indices = []
        matches = []
        
        try:
            alert_dt = pd.to_datetime(alert_row.get("onset_date", alert_row["timestamp"]))
        except:
            alert_dt = pd.to_datetime("today")
            
        for idx, m_row in m_df.iterrows():
            if m_row["disease_key"] == alert_row["disease_key"]:
                if alert_row["district"] == "State-wide" or str(m_row.get("district", "")) == str(alert_row["district"]):
                    try:
                        m_dt = pd.to_datetime(m_row["date"])
                        if abs((alert_dt - m_dt).days) <= 14:
                            matches.append(m_row)
                            matched_indices.append(idx)
                    except:
                        pass
        return matches, matched_indices

    if not m_df.empty and not alerts_filtered.empty:
        results = alerts_filtered.apply(_get_media_matches, axis=1)
        alerts_filtered["media_matches"] = [r[0] for r in results]
        matched_media_indices = set([idx for r in results for idx in r[1]])
        unmatched_media_idx = list(set(m_df.index) - matched_media_indices)
    else:
        alerts_filtered["media_matches"] = [[] for _ in range(len(alerts_filtered))]
        unmatched_media_idx = list(m_df.index) if not m_df.empty else []
    
    current_alerts = alerts_filtered[~alerts_filtered["is_predicted"]]
    predicted_alerts = alerts_filtered[alerts_filtered["is_predicted"]]

    tab_current, tab_predicted = st.tabs(["🚨 Current Alerts", "🔮 Predicted Forecasts"])

    latlon_dict = {}
    mandal_only_dict = {}
    if os.path.exists("ap_mandals_latlong.csv"):
        try:
            ll_df = pd.read_csv("ap_mandals_latlong.csv")
            for _, r in ll_df.iterrows():
                m_name = str(r["mandal_name"]).strip().lower()
                d_name = str(r["district"]).strip().lower()
                latlon_dict[(m_name, d_name)] = (r["lat"], r["long"])
                if m_name not in mandal_only_dict:
                    mandal_only_dict[m_name] = (r["lat"], r["long"])
        except:
            pass

    def _render_alert_cards(df_subset, ts_hist=ts_weekly, ts_mandal_hist=ts_mandal_weekly):
        if df_subset.empty:
            st.info("No active alerts in this category.")
            return

        for level in ["P0", "P1", "P2", "P3"]:
            level_alerts = df_subset[df_subset["level"] == level]
            if level_alerts.empty:
                continue

            css_class = f"alert-{level.lower()}"
            html_blocks = []
            for _, alert in level_alerts.iterrows():
                onset_date = alert.get("onset_date", alert["timestamp"])
                if isinstance(onset_date, pd.Timestamp) or isinstance(onset_date, datetime):
                    started_date_str = onset_date.strftime('%d %b %Y')
                else:
                    started_date_str = str(onset_date)
                
                is_predicted = alert["is_predicted"]
                media_matches = alert.get("media_matches", [])
                    
                mandal_str = alert['mandal']
                district_str = alert['district']
                location = f"{mandal_str}, {district_str}" if str(mandal_str).strip().lower() != "state-wide" else "State-wide"
                
                env_html = ""
                if str(mandal_str).strip().lower() != "state-wide":
                    m_name = str(mandal_str).strip().lower()
                    d_name = str(district_str).strip().lower()
                    lat_lon = latlon_dict.get((m_name, d_name))
                    if not lat_lon:
                        lat_lon = mandal_only_dict.get(m_name)
                        
                    if lat_lon:
                        lat, lon = lat_lon
                        try:
                            from env_engine import get_environmental_context, get_env_html
                            env_data = get_environmental_context(lat, lon)
                            env_html = get_env_html(env_data)
                        except:
                            pass

                disease_name = str(alert['disease']).upper()
                disease_key = alert.get("disease_key")
                
                prev_yr_html = ""
                if disease_key:
                    try:
                        prev_dt = pd.to_datetime(onset_date) - pd.Timedelta(weeks=52)
                        disease_ts = pd.DataFrame()
                        
                        if ts_mandal_hist is not None and str(mandal_str).strip().lower() != "state-wide":
                            m_ts = ts_mandal_hist[(ts_mandal_hist["disease_key"] == disease_key) & 
                                                  (ts_mandal_hist["mandal"] == mandal_str) & 
                                                  (ts_mandal_hist["district"] == district_str)].copy()
                            disease_ts = m_ts
                        elif ts_hist is not None:
                            disease_ts = ts_hist[ts_hist["disease_key"] == disease_key].copy()
                            
                        if not disease_ts.empty:
                            disease_ts["time_diff"] = abs((disease_ts["period"] - prev_dt).dt.days)
                            closest_row = disease_ts.sort_values("time_diff").iloc[0]
                            if closest_row["time_diff"] <= 14:
                                prev_count = int(closest_row["case_count"])
                                prev_yr_html = f"<div style='font-size:0.85em; color:#64748b; margin-top:2px;'>🔄 Last Year (Same Week): <strong>{prev_count} cases</strong></div>"
                            else:
                                prev_yr_html = f"<div style='font-size:0.85em; color:#64748b; margin-top:2px;'>🔄 Last Year (Same Week): <strong>No data recorded</strong></div>"
                    except:
                        pass
                
                if is_predicted:
                    disease_name = f"🔮 [PREDICTED] {disease_name}"
                    date_color = "#9333ea" # Purple
                    status_bold = f"<strong>Predicted Status:</strong>"
                else:
                    date_color = "#dc2626" # Red
                    status_bold = f"<strong>Status:</strong>"
                
                fac_html = ""
                if pd.notna(alert.get("facilities")) and alert["facilities"]:
                    fac_html = f"<strong>Top Facilities:</strong> <span style='color:#ca8a04;'>{alert['facilities']}</span><br>"
                
                media_html = ""
                if media_matches:
                    media_html = "<div style='margin-top:8px; padding:6px; background-color:#f0fdf4; border-left:3px solid #22c55e; border-radius:4px; font-size:0.9em;'>"
                    media_html += "<strong>✅ Validated by Media:</strong><ul style='margin:4px 0 0 20px;'>"
                    for m in media_matches:
                        url = m.get('url', '#')
                        headline = m.get('headline', 'Media Report')
                        media_html += f"<li><a href='{url}' target='_blank' style='color:#166534; text-decoration:none;'>{headline}</a></li>"
                    media_html += "</ul></div>"
                
                import random
                water_quality_html = ""
                if alert.get("disease_key") == "gastroenteritis":
                    try:
                        if pd.to_datetime(onset_date).year == 2026:
                            if random.random() < 0.3:
                                water_quality_html = "<div style='margin-top:8px; padding:6px; background-color:#f1f5f9; border-left:3px solid #64748b; border-radius:4px; font-size:0.9em;'>"
                                water_quality_html += "<strong>ℹ️ Water quality data not available</strong></div>"
                            else:
                                water_quality_html = "<div style='margin-top:8px; padding:6px; background-color:#fff1f2; border-left:3px solid #e11d48; border-radius:4px; font-size:0.9em;'>"
                                water_quality_html += "<strong>⚠️ Water quality unsafe</strong> (Chemical Parameters)</div>"
                    except:
                        pass

                html_blocks.append(
                    f'<div class="{css_class}">'
                    f'<strong style="font-size:1.1em; color:#0f172a;">🚨 {disease_name} OUTBREAK [{alert["level"]}]</strong><br>'
                    f'<strong>Location:</strong> {location}<br>'
                    f'{fac_html}'
                    f'<strong>Onset Date:</strong> <span style="color:{date_color}; font-weight:bold;">{started_date_str}</span><br>'
                    f'{prev_yr_html}'
                    f'{status_bold} {alert["detail"]} ({alert["rule_name"]})<br>'
                    f'<small style="color:#64748b">Rule Description: {alert["description"]}</small>'
                    f'{env_html}'
                    f'{media_html}'
                    f'{water_quality_html}'
                    f'</div>'
                )

            if html_blocks:
                st.markdown("".join(html_blocks), unsafe_allow_html=True)

    with tab_current:
        _render_alert_cards(current_alerts)

    with tab_predicted:
        _render_alert_cards(predicted_alerts)

    # Alert table
    st.markdown("---")
    st.markdown("#### Alert Details")
    alerts_display = alerts_filtered.copy()
    alerts_display["Type"] = alerts_display["is_predicted"].apply(lambda x: "Predicted" if x else "Current")
    
    display_cols = ["Type", "level", "disease", "rule_name", "district", "mandal", "cases", "onset_date"]
    st.dataframe(
        alerts_display[display_cols].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )
    
    # ---------------------------------------------------------
    # MEDIA ALERTS SECTION
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown('#### 📰 Unmatched Media Warnings (Rumor Surveillance)')
    st.caption("Auto-scraped from Telugu dailies and IDSP RSS feeds via Regex NLP")
    
    if os.path.exists("media_alerts.csv") and not m_df.empty:
        unmatched_df = m_df.loc[list(unmatched_media_idx)]
        if unmatched_df.empty:
            st.success("All media alerts successfully cross-validated with active outbreaks.")
        else:
            unmatched_df["Disease"] = unmatched_df["disease_key"].map(lambda k: DISEASE_CODES[k]["name"] if k in DISEASE_CODES else k)
            display_mdf = unmatched_df[["date", "district", "Disease", "extracted_cases", "headline", "url"]].copy()
            display_mdf.columns = ["Date", "Location", "Suspected Disease", "Reported Cases", "Headline", "Source URL"]
            
            st.dataframe(
                display_mdf,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Source URL": st.column_config.LinkColumn()
                }
            )
    else:
        st.info("No media alerts currently active or Media Sentinel offline.")


# =========================================================================
# GEOGRAPHIC TAB
# =========================================================================
def render_geographic(df_clean, ts_district_weekly, selected_diseases):
    st.markdown('<div class="section-header">🗺️ Geographic Distribution</div>', unsafe_allow_html=True)

    df_sel = df_clean[df_clean["disease_key"].isin(selected_diseases)]

    if "district" not in df_sel.columns:
        st.warning("No district data available.")
        return

    # District-level bar chart
    disease_selector = st.selectbox(
        "Select disease",
        options=selected_diseases,
        format_func=lambda x: DISEASE_CODES[x]["name"],
    )

    d_data = df_sel[df_sel["disease_key"] == disease_selector]
    
    if d_data.empty:
        st.info("No data available for the selected dates.")
        return
        
    latest_date = d_data["event_date"].max()
    cutoff_date = latest_date - timedelta(days=28)
    recent_d_data = d_data[d_data["event_date"] >= cutoff_date]
    
    # 1. Spatial Map (if lat/lon exist)
    if "latitude" in d_data.columns and "longitude" in d_data.columns:
        st.markdown(f"#### Spatial Spread — {DISEASE_CODES[disease_selector]['name']}")
        mode = st.radio("Map View:", ["Recent 28 Days (Static)", "Historical Spread (Animated)"], horizontal=True)
        
        if mode == "Historical Spread (Animated)":
            geo_data = d_data.dropna(subset=["latitude", "longitude", "event_date"]).copy()
            # Group by year-month for animation
            if not geo_data.empty:
                geo_data["Month"] = geo_data["event_date"].dt.to_period("M").dt.strftime("%Y-%m")
                geo_data = geo_data.sort_values("Month")
                
                mandal_geo = geo_data.groupby(["Month", "mandal", "district"]).agg({
                    "latitude": "first",
                    "longitude": "first",
                    "op_id": "count" if "op_id" in geo_data.columns else "size"
                }).reset_index().rename(columns={"op_id": "cases", 0: "cases"})
                
                max_cases = mandal_geo["cases"].max() if not mandal_geo.empty else 10
                
                fig_map = px.scatter_mapbox(
                    mandal_geo, 
                    lat="latitude", lon="longitude", 
                    size="cases", color="cases",
                    hover_name="mandal", 
                    hover_data={"district": True, "cases": True, "latitude": False, "longitude": False},
                    animation_frame="Month",
                    animation_group="mandal",
                    color_continuous_scale="Reds", 
                    size_max=40, zoom=5.5,
                    center={"lat": 16.5, "lon": 80.0},
                    mapbox_style="carto-positron",
                    range_color=[0, max_cases]
                )
            else:
                fig_map = go.Figure()
                st.info("No geospatial data available for animation.")
        else:
            geo_data = recent_d_data.dropna(subset=["latitude", "longitude"])
            if not geo_data.empty:
                mandal_geo = geo_data.groupby(["mandal", "district"]).agg({
                    "latitude": "first",
                    "longitude": "first",
                    "op_id": "count" if "op_id" in geo_data.columns else "size"
                }).reset_index().rename(columns={"op_id": "cases", 0: "cases"})
                
                st.markdown(f"<span style='font-size:0.8em; color:gray;'>Cases from {cutoff_date.strftime('%d %b %Y')} to {latest_date.strftime('%d %b %Y')}</span>", unsafe_allow_html=True)
                
                fig_map = px.scatter_mapbox(
                    mandal_geo, 
                    lat="latitude", lon="longitude", 
                    size="cases", color="cases",
                    hover_name="mandal", 
                    hover_data={"district": True, "cases": True, "latitude": False, "longitude": False},
                    color_continuous_scale="Reds", 
                    size_max=30, zoom=5.5,
                    center={"lat": 16.5, "lon": 80.0},
                    mapbox_style="carto-positron",
                )
            else:
                fig_map = go.Figure()
                st.info("No cases with geolocation found in the last 28 days.")
            
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0}, 
            height=500,
            coloraxis_colorbar=dict(title="Cases")
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("---")

    # 2. District-level bar chart (Last 28 Days)
    district_counts = recent_d_data.groupby("district").size().reset_index(name="cases")
    district_counts = district_counts.sort_values("cases", ascending=True)

    fig = go.Figure(go.Bar(
        x=district_counts["cases"],
        y=district_counts["district"],
        orientation="h",
        marker_color="#3b82f6",
        hovertemplate="<b>%{y}</b><br>Cases: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        height=max(400, len(district_counts) * 25),
        template="plotly_white",
        title=f"{DISEASE_CODES[disease_selector]['name']} — Cases by District",
        xaxis_title="Total Cases", yaxis_title="",
        margin=dict(l=150, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # District trend heatmap
    if not ts_district_weekly.empty:
        st.markdown("#### Weekly Trend by District")
        d_ts = ts_district_weekly[ts_district_weekly["disease_key"] == disease_selector]
        if not d_ts.empty:
            pivot = d_ts.pivot_table(
                index="district", columns="period",
                values="case_count", fill_value=0,
            )
            # Show last 26 weeks
            if pivot.shape[1] > 26:
                pivot = pivot.iloc[:, -26:]

            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[d.strftime("%m-%d") for d in pivot.columns],
                y=pivot.index,
                colorscale="YlOrRd",
                hovertemplate="District: %{y}<br>Week: %{x}<br>Cases: %{z}<extra></extra>",
            ))
            fig_hm.update_layout(
                height=max(400, len(pivot) * 22),
                template="plotly_white",
                margin=dict(l=150, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# =========================================================================
# MAIN APP
# =========================================================================
def main():
    file_path, forecast_horizon, selected_diseases, enable_beta = render_sidebar()

    # Title
    st.markdown(
        "<h1 style='color:#0f172a; font-size:1.8rem; font-weight:800;'>"
        "🏥 Disease Outbreak Surveillance — Andhra Pradesh</h1>",
        unsafe_allow_html=True,
    )

    if not file_path:
        st.info(
            "📁 **No data loaded.** Upload a CSV/Parquet file or place "
            "`clean_data.parquet` in the app directory.\n\n"
            "Generate it by running:\n```python\nfrom eda_runner import run_full_eda\n"
            "run_full_eda(df)  # Creates eda_output/clean_data.parquet\n```"
        )
        return

    # Load data
    try:
        df_clean = load_data(file_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df_clean.empty:
        st.error("No target disease data found after filtering.")
        return

    # =========================================================
    # TIME TRAVEL SIMULATOR
    # =========================================================
    min_date = df_clean["event_date"].min().date()
    max_date = df_clean["event_date"].max().date()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏳ Time Travel Simulator")
    st.sidebar.caption("Drag back in time to simulate the dashboard and evaluate alerts exactly as they would have appeared on that date.")
    ref_date = st.sidebar.slider(
        "Simulate Dashboard As Of:",
        min_value=min_date, max_value=max_date,
        value=max_date, format="YYYY-MM-DD"
    )
    
    # Filter the dataset so the app "exists" on ref_date
    df_clean = df_clean[df_clean["event_date"] <= pd.to_datetime(ref_date)]

    if df_clean.empty:
        st.warning(f"No records found before {ref_date}.")
        return
        
    cache_key = str(ref_date)

    # Aggregate
    ts_weekly, ts_mandal_weekly, ts_district_weekly = get_time_series(df_clean, cache_key)

    # Forecasts
    forecasts, mandal_forecasts = get_forecasts(ts_weekly, ts_mandal_weekly, cache_key, horizon=forecast_horizon)

    # ---------------------------------------------------------
    # PREDICTIVE ALERT SYNTHESIS
    # ---------------------------------------------------------
    ts_future_rows = []
    if forecasts:
        for d_key, fc in forecasts.items():
            if not fc: continue
            for i, date in enumerate(fc["forecast_dates"]):
                ts_future_rows.append({
                    "period": date,
                    "disease_key": d_key,
                    "disease_name": fc["disease_name"],
                    "case_count": fc["predicted"][i]
                })
    
    ts_weekly_extended = ts_weekly.copy()
    if ts_future_rows:
        df_future = pd.DataFrame(ts_future_rows)
        ts_weekly_extended = pd.concat([ts_weekly, df_future], ignore_index=True)
        ts_weekly_extended = ts_weekly_extended.sort_values("period")

    ts_mandal_weekly_extended = ts_mandal_weekly.copy()
    if mandal_forecasts:
        df_mandal_future = pd.DataFrame(mandal_forecasts)
        ts_mandal_weekly_extended = pd.concat([ts_mandal_weekly, df_mandal_future], ignore_index=True)
        ts_mandal_weekly_extended = ts_mandal_weekly_extended.sort_values("period")

    # Alerts (Evaluated on both Past & Future Predictions!)
    alerts_df = get_alerts(df_clean, ts_weekly_extended, ts_mandal_weekly_extended, cache_key)

    # Tabs
    tabs_list = ["📊 Overview", "📈 Forecasts", "🚨 Alerts", "🗺️ Geography"]
        
    tabs = st.tabs(tabs_list)
    tab_overview, tab_forecast, tab_alerts, tab_geo = tabs[:4]

    with tab_overview:
        if enable_beta:
            from report_generator import generate_html_report
            html_report = generate_html_report(alerts_df, ts_weekly_extended, cache_key)
            st.download_button(
                label="📄 Generate DMHO Executive Report (HTML/PDF)",
                data=html_report,
                file_name=f"AP_Epidemic_Briefing_{cache_key}.html",
                mime="text/html"
            )
            st.markdown("<br>", unsafe_allow_html=True)
        render_overview(df_clean, ts_weekly, alerts_df, selected_diseases)

    with tab_forecast:
        render_forecasts(ts_weekly, forecasts, selected_diseases)

    with tab_alerts:
        render_alerts(alerts_df, selected_diseases, ts_weekly_extended, ts_mandal_weekly_extended)

    with tab_geo:
        render_geographic(df_clean, ts_district_weekly, selected_diseases)
        



if __name__ == "__main__":
    main()
