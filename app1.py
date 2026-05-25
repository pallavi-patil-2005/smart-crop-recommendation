"""
=============================================================================
  🌾 CROP RECOMMENDATION SYSTEM — FastAPI Backend
=============================================================================
SETUP:
    pip install fastapi uvicorn pandas scikit-learn joblib

FILES NEEDED in same directory:
    model1_npk.pkl
    model1_label_encoder.pkl
    model2_full_scored.csv
    index.html  (the frontend file)

RUN:
    uvicorn main:app --reload --port 8000
    -> Opens at http://localhost:8000
=============================================================================
"""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

app = FastAPI(title="Crop Recommendation System")

# -----------------------------------------------------------------------------
# LOAD MODELS  (graceful fallback if files missing — for dev/demo)
# -----------------------------------------------------------------------------
try:
    model1     = joblib.load("model1_npk.pkl")
    label_enc1 = joblib.load("model1_label_encoder.pkl")
    MODEL1_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load model1: {e}")
    MODEL1_LOADED = False

try:
    model2_df = pd.read_csv("model2_full_scored.csv")
    model2_df["District"] = model2_df["District"].str.strip()
    model2_df["District"] = model2_df["District"].apply(
        lambda x: x[0].upper() + x[1:] if isinstance(x, str) and len(x) > 0 else x
    )
    model2_df = model2_df[
        model2_df["District"].notna() &
        (model2_df["District"].str.strip() != "")
    ]
    DISTRICTS = sorted(model2_df["District"].unique().tolist())
    SEASONS   = sorted(model2_df["Season"].unique().tolist())
    MODEL2_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load model2 CSV: {e}")
    model2_df = pd.DataFrame()
    DISTRICTS = ["Bagalkot", "Bangalore Rural", "Bangalore Urban", "Belgaum",
                 "Bellary", "Bidar", "Bijapur", "Chamarajanagar", "Chikballapur",
                 "Chikmagalur", "Chitradurga", "Dakshina Kannada", "Davangere",
                 "Dharwad", "Gadag", "Gulbarga", "Hassan", "Haveri", "Kodagu",
                 "Kolar", "Koppal", "Mandya", "Mysore", "Raichur", "Ramanagara",
                 "Shimoga", "Tumkur", "Udupi", "Uttara Kannada", "Yadgir"]
    SEASONS   = ["Kharif", "Rabi", "Summer", "Whole Year"]
    MODEL2_LOADED = False

def norm(name):
    return (str(name).strip().lower()
            .replace(" ", "").replace("(", "").replace(")", "")
            .replace("-", "").replace("/", "").replace("&", ""))

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def model1_scores(N, P, K, temperature, humidity, ph, rainfall):
    if not MODEL1_LOADED:
        return {}
    X = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )
    proba = model1.predict_proba(X)[0]
    return {c: float(p) for c, p in zip(label_enc1.classes_, proba)}

def model2_scores(district, season):
    if not MODEL2_LOADED:
        return {}
    region = model2_df[
        (model2_df["District"] == district) &
        (model2_df["Season"]   == season)
    ]
    if region.empty:
        return {}
    return dict(zip(region["Crop"], region["Suitability_Score"].astype(float)))

def recommend(N, P, K, temperature, humidity, ph, rainfall,
              district, season, w_model1, top_n):
    w_model2 = round(1 - w_model1, 2)
    m1 = model1_scores(N, P, K, temperature, humidity, ph, rainfall)
    m2 = model2_scores(district, season)

    m1_norm_map = {norm(k): (k, v) for k, v in m1.items()}
    m2_norm_map = {norm(k): (k, v) for k, v in m2.items()}
    all_keys    = set(m1_norm_map) | set(m2_norm_map)

    rows = []
    for key in all_keys:
        _, s1 = m1_norm_map.get(key, (key, 0.0))
        crop_display, s2 = m2_norm_map.get(key, (key, 0.0))
        if key not in m2_norm_map:
            crop_display = m1_norm_map[key][0].title()

        combined = round(w_model1 * s1 + w_model2 * s2, 4)

        reg_label = hist_grown = avg_yield = freq_pct = None
        if MODEL2_LOADED:
            region_row = model2_df[
                (model2_df["District"] == district) &
                (model2_df["Season"]   == season) &
                (model2_df["Crop"].apply(norm) == key)
            ]
            reg_label  = region_row["Recommendation"].values[0]             if not region_row.empty else "No regional data"
            hist_grown = int(region_row["Is_Historically_Grown"].values[0]) if not region_row.empty else 0
            avg_yield  = round(float(region_row["Avg_Yield"].values[0]), 2) if not region_row.empty else 0.0
            freq_pct   = round(float(region_row["Crop_Freq_Pct"].values[0]), 1) if not region_row.empty else 0.0

        rows.append({
            "crop"           : crop_display,
            "confidence"     : round(combined * 100, 1),
            "npk_score"      : round(s1 * 100, 1),
            "region_score"   : round(s2 * 100, 1),
            "suitability"    : reg_label or "No regional data",
            "freq_pct"       : freq_pct or 0.0,
            "avg_yield"      : avg_yield or 0.0,
            "hist_grown"     : "Yes" if hist_grown else "No",
            "combined"       : combined,
        })

    rows.sort(key=lambda x: x["combined"], reverse=True)
    rows = rows[:int(top_n)]
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        del r["combined"]
    return rows

# -----------------------------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/meta")
async def get_meta():
    return {"districts": DISTRICTS, "seasons": SEASONS}

@app.get("/api/recommend")
async def api_recommend(
    N: float = 90, P: float = 42, K: float = 43,
    temperature: float = 25, humidity: float = 80,
    ph: float = 6.5, rainfall: float = 200,
    district: str = "Bagalkot", season: str = "Kharif",
    w_model1: float = 0.6, top_n: int = 5
):
    try:
        result = recommend(N, P, K, temperature, humidity, ph, rainfall,
                           district, season, w_model1, top_n)
        return {"status": "ok", "results": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/browse")
async def api_browse(
    district: str = "Bagalkot",
    season: str = "Kharif",
    top_n: int = 8
):
    if not MODEL2_LOADED:
        return {"status": "ok", "results": []}
    try:
        region = model2_df[
            (model2_df["District"] == district) &
            (model2_df["Season"]   == season)
        ].sort_values("Suitability_Score", ascending=False).head(int(top_n))

        results = []
        for i, (_, row) in enumerate(region.iterrows()):
            results.append({
                "rank"        : i + 1,
                "crop"        : row["Crop"],
                "score_pct"   : round(float(row["Suitability_Pct"]), 1),
                "suitability" : row["Recommendation"],
                "freq_pct"    : round(float(row["Crop_Freq_Pct"]), 1),
                "avg_yield"   : round(float(row["Avg_Yield"]), 2),
                "dominance"   : round(float(row["Dominance_Rate"]), 1),
                "recent_years": int(row["Recent_Years_Grown"]),
            })
        return {"status": "ok", "results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("\n🌾 Starting Crop Recommendation System — Karnataka")
    print(f"-> Districts loaded : {len(DISTRICTS)}")
    print("-> Open             : http://localhost:8000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
