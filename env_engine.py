"""
Environmental Context Engine
============================
Fetches real-time weather and air quality data to provide risk context to outbreaks.
Uses Open-Meteo API (No API Key required).
"""
import requests
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600, show_spinner=False)
def get_environmental_context(lat, lon):
    """
    Fetches Weather and AQI data from Open-Meteo.
    Caches the result for 1 hour to prevent API throttling and ensure fast dashboard rendering.
    """
    if pd.isna(lat) or pd.isna(lon):
        return None
        
    try:
        # Weather with past 3 days of precipitation
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&daily=precipitation_sum&past_days=3&forecast_days=1"
        w_res = requests.get(w_url, timeout=1.5).json()
        temp = w_res.get("current", {}).get("temperature_2m")
        hum = w_res.get("current", {}).get("relative_humidity_2m")
        
        # Calculate recent rainfall accumulation
        daily_precip = w_res.get("daily", {}).get("precipitation_sum", [])
        total_recent_rain = sum([p for p in daily_precip if p is not None])
        
        # AQI
        a_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,european_aqi"
        a_res = requests.get(a_url, timeout=1.5).json()
        aqi = a_res.get("current", {}).get("european_aqi")
        pm25 = a_res.get("current", {}).get("pm2_5")
        
        return {
            "temp": temp,
            "humidity": hum,
            "recent_rain": total_recent_rain,
            "aqi": aqi,
            "pm25": pm25
        }
    except Exception as e:
        return None

def get_env_html(env_data):
    if not env_data:
        return ""
    
    t = env_data.get("temp")
    h = env_data.get("humidity")
    r = env_data.get("recent_rain", 0)
    a = env_data.get("aqi")
    
    if t is None or h is None:
        return ""
        
    # Analyze risk (Mosquito breeding needs standing water + humidity)
    risk_notes = []
    if (h and h >= 70) and (r > 3.0):
        risk_notes.append(f"High Mosquito Breeding Risk ({r:.1f}mm Rain + Humidity)")
    elif r > 10.0:
        risk_notes.append(f"Waterborne Disease Risk ({r:.1f}mm Heavy Rain)")
        
    if a and a > 100:
        risk_notes.append("Poor Air Quality (Resp. Risk)")
        
    warning_color = "#991b1b" if risk_notes else "#0f766e"
    bg_color = "#fee2e2" if risk_notes else "#f0fdfa"
    
    risk_str = f" - <strong style='color:{warning_color};'><em>{', '.join(risk_notes)}</em></strong>" if risk_notes else ""
    aqi_str = f" | 🌫️ AQI: {a}" if a is not None else ""
    
    return f"<div style='font-size:0.85em; color:#0f766e; margin-top:4px; padding:6px; background:{bg_color}; border-radius:4px; border-left: 3px solid {warning_color};'><strong>🌍 Live Env:</strong> 🌡️ {t}°C | 💧 Hum: {h}%{aqi_str}{risk_str}</div>"
