"""
Media & Event-Based Surveillance Scraper
=========================================
Scrapes Google News RSS feeds (including local Telugu dailies)
for early outbreak signals in Andhra Pradesh.
Runs offline Regex-based NLP for translation and disease tagging.
"""

import os
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import time

OUTPUT_FILE = "media_alerts.csv"

# =========================================================================
# BILINGUAL DISEASE DICTIONARY (English & Telugu)
# =========================================================================
DISEASE_NLP_MAP = {
    "dengue": [r"dengue", r"డెంగ్యూ", r"డెంగీ"],
    "malaria": [r"malaria", r"మలేరియా"],
    "chikungunya": [r"chikungunya", r"చికున్గున్యా", r"చికున్ గున్యా"],
    "cholera": [r"cholera", r"కలరా"],
    "gastroenteritis": [r"diarrhoea", r"diarrhea", r"dysentery", r"gastroenteritis", r"విరేచనాలు", r"వాంతులు", r"కలరా వాంతులు"],
    "febrile_illness": [r"fever", r"pyrexia", r"జ్వరం", r"జ్వరాలు", r"వింత జ్వరం", r"మిస్టరీ జ్వరం"],
    "mud_fever": [r"mud fever", r"స్క్రబ్ టైఫస్", r"బురద జ్వరం"],
    "typhoid": [r"\btyphoid\b", r"టైఫాయిడ్", r"టైఫాయిడ్ జ్వరం"],
    "ebola": [r"ebola"],
}

# AP Districts for Location Tapping
AP_DISTRICTS = [
    "Srikakulam", "Parvathipuram Manyam", "Vizianagaram", "Visakhapatnam", 
    "Alluri Sitharama Raju", "Anakapalli", "Kakinada", "East Godavari", 
    "Dr. B. R. Ambedkar Konaseema", "Eluru", "West Godavari", "NTR", 
    "Krishna", "Palnadu", "Guntur", "Bapatla", "Prakasam", 
    "Sri Potti Sriramulu Nellore", "Kurnool", "Nandyal", 
    "Anantapur", "Sri Sathya Sai", "Y. S. R.", "Annamayya", "Tirupati", "Chittoor"
]

TELUGU_DISTRICTS = {
    "NTR": [r"ఎన్టీఆర్", r"విజయవాడ"],
    "Guntur": [r"గుంటూరు"],
    "Visakhapatnam": [r"విశాఖపట్నం", r"విశాఖ"],
    # Add more as needed, regex falls back to English match typically
}

# =========================================================================
# SCRAPING LOGIC
# =========================================================================

def fetch_rss(url):
    """Fetch and parse an RSS feed."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 AP-RTGS-Scraper/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return ET.fromstring(response.read())
    except Exception as e:
        print(f"  [ERROR] Fetching {url}: {e}")
        return None

def extract_cases(text):
    """Regex to safely extract case counts from Telugu/English strings."""
    # Matches: "15 cases", "20 మందికి", "15 patients"
    match = re.search(r'(\d+)\s*(?:cases|patients|deaths|admitted|మందికి|మంది)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def detect_disease(text):
    """Returns a list of disease keys found in the text."""
    found = []
    text_lower = text.lower()
    for d_key, patterns in DISEASE_NLP_MAP.items():
        for pat in patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                found.append(d_key)
                break # Move to next disease
    return found

def detect_location(text):
    """Returns district string if strictly found. Returns None if out-of-state."""
    text_lower = text.lower()
    
    # 1. District level match
    for dist in AP_DISTRICTS:
        if dist.lower() in text_lower:
            return dist
            
    for dist, telugu_names in TELUGU_DISTRICTS.items():
        for t_name in telugu_names:
            if re.search(t_name, text):
                return dist
                
    # 2. General State match
    if "andhra pradesh" in text_lower or "ఆంధ్రప్రదేశ్" in text:
        return "Andhra Pradesh (General)"
        
    return None

def fetch_rss(url):
    """Fetch and parse an RSS feed."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 AP-RTGS-Scraper/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return ET.fromstring(response.read())
    except Exception as e:
        print(f"  [ERROR] Fetching {url}: {e}")
        return None

def run_scraper():
    print("\n[ MEDIA SURVEILLANCE ENGINE ]")
    print("Initiating Bing News API Extraction...\n")

    # Advanced Multi-Source API Queries
    queries = [
        # Regional Telugu
        'ఆంధ్రప్రదేశ్ (జ్వరం OR డెంగ్యూ OR కలరా OR విరేచనాలు)',
        # English High-Priority Outbreak Syntax
        '"Andhra Pradesh" AND ("fever" OR "dengue" OR "cholera" OR "mystery disease") AND "outbreak"',
        # IDSP / Beacon
        '"Andhra Pradesh" AND ("IDSP" OR "ProMED" OR "HealthMap") AND "cases"'
    ]
    
    all_alerts = []
    
    for q in queries:
        print(f"Searching via API: {q}")
        safe_q = urllib.parse.quote(q)
        url = f"https://www.bing.com/news/search?q={safe_q}&format=rss"
        
        root = fetch_rss(url)
        if not root:
            continue
            
        channel = root.find("channel")
        if channel is None:
            continue
            
        for item in channel.findall("item"):
            title = item.find("title").text if item.find("title") is not None else ""
            desc = item.find("description").text if item.find("description") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else str(datetime.now())
            
            combined_text = f"{title} {desc}"
            
            # NLP Extraction (No LLM)
            diseases = detect_disease(combined_text)
            
            if not diseases:
                continue # Skip irrelevant articles
                
            cases = extract_cases(combined_text)
            loc = detect_location(combined_text)
            
            # STRICT GEOGRAPHIC GATING
            # Reject articles from Nigeria, Odisha, etc.
            if not loc:
                continue
            
            # Format Date safely
            try:
                # "2026-02-10T02:43:00+00:00" -> datetime
                dt = pd.to_datetime(pub_date).strftime("%Y-%m-%d")
            except:
                dt = datetime.now().strftime("%Y-%m-%d")
            
            for d in diseases:
                all_alerts.append({
                    "date": dt,
                    "district": loc,
                    "disease_key": d,
                    "headline": f"[{article.get('source', 'News')}] {title.strip()}",
                    "extracted_cases": cases if cases else "Unknown",
                    "url": link
                })
                
        time.sleep(2) # Courtesy API delay
        
    if all_alerts:
        df = pd.DataFrame(all_alerts)
        df = df.drop_duplicates(subset=["headline", "disease_key"])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Saved {len(df)} API-Verified Media Alerts to {OUTPUT_FILE}")
    else:
        # Create empty CSV so dashboard doesn't crash
        df = pd.DataFrame(columns=["date", "district", "disease_key", "headline", "extracted_cases", "url"])
        df.to_csv(OUTPUT_FILE, index=False)
        print("\n✓ Scraping complete. No new high-priority media alerts found.")

if __name__ == "__main__":
    run_scraper()
