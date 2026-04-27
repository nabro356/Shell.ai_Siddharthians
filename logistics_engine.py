"""
Auto-Logistics & Supply Predictor
==================================
Maps disease forecasts directly to physical supply chain requirements
to prevent hospital stockouts before they happen.
"""

import pandas as pd

# Heuristics for supply calculation based on 1 Case of a specific disease
DISEASE_LOGISTICS_MAP = {
    "dengue": {
        "IV Fluids (Liters)": 2.0,
        "Platelet Units": 0.2,
        "Hospital Beds": 0.15,
        "Paracetamol (Strips)": 1.0
    },
    "malaria": {
        "ACT Doses": 1.0,
        "Rapid Diagnostic Tests (RDT)": 2.0,
        "Mosquito Nets": 0.5,
        "Hospital Beds": 0.05
    },
    "gastroenteritis": {
        "ORS Packets": 5.0,
        "IV Fluids (Liters)": 1.5,
        "Antibiotic Courses": 0.3,
        "Hospital Beds": 0.02
    },
    "typhoid": {
        "Antibiotic Courses": 1.0,
        "Widal Test Kits": 1.5,
        "Hospital Beds": 0.1
    },
    "chikungunya": {
        "Painkillers (Strips)": 2.0,
        "IV Fluids (Liters)": 0.5
    },
    "cholera": {
        "ORS Packets": 10.0,
        "IV Fluids (Liters)": 5.0,
        "Antibiotic Courses": 1.0,
        "Hospital Beds": 0.2
    },
    # Default fallback for unmapped diseases
    "default": {
        "General Consultations": 1.0,
        "Basic Medical Kits": 0.5
    }
}

def calculate_logistics(mandal_forecasts):
    """
    Takes the mandal-level forecasts (which contain 'disease_key', 'mandal', 'district', 'case_count')
    and translates them into a supply chain demand dataframe.
    """
    if not mandal_forecasts:
        return pd.DataFrame()
        
    df_fcst = pd.DataFrame(mandal_forecasts)
    if df_fcst.empty:
        return pd.DataFrame()
        
    # Aggregate total cases predicted per district and disease over the entire forecast horizon
    district_totals = df_fcst.groupby(["district", "disease_key", "disease_name"])["case_count"].sum().reset_index()
    
    logistics_rows = []
    for _, row in district_totals.iterrows():
        d_key = row["disease_key"]
        d_name = row["disease_name"]
        district = row["district"]
        cases_raw = row["case_count"]
        if pd.isna(cases_raw) or cases_raw == float('inf') or cases_raw == float('-inf'):
            cases = 0
        else:
            # Cap maximum cases at 1,000,000 per mandal 
            # to prevent PyArrow Int64 overflow on absurd ML extrapolations
            cases = min(1_000_000, max(0, int(cases_raw)))
        
        if cases == 0:
            continue
            
        mapping = DISEASE_LOGISTICS_MAP.get(d_key, DISEASE_LOGISTICS_MAP["default"])
        
        for item, multiplier in mapping.items():
            required_qty = int(cases * multiplier)
            if required_qty > 0:
                logistics_rows.append({
                    "District": district,
                    "Disease": d_name,
                    "Predicted Cases": cases,
                    "Required Item": item,
                    "Estimated Quantity": required_qty
                })
                
    if not logistics_rows:
        return pd.DataFrame()
        
    log_df = pd.DataFrame(logistics_rows)
    # Group by district and item to get total district demands across all diseases
    summary_df = log_df.groupby(["District", "Required Item"])["Estimated Quantity"].sum().reset_index()
    summary_df = summary_df.sort_values(by=["District", "Estimated Quantity"], ascending=[True, False])
    
    return log_df, summary_df
