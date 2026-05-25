"""
=============================================================================
  🌾 CROP RECOMMENDATION SYSTEM — FastAPI Backend (Enhanced)
=============================================================================
SETUP:
    pip install fastapi uvicorn pandas scikit-learn joblib pyngrok

FILES NEEDED in same directory:
    model1_npk.pkl
    model1_label_encoder.pkl
    model2_full_scored.csv
    index.html  (the frontend file)

RUN:
    python main.py
    -> Opens locally at http://localhost:8000
    -> Opens publicly via ngrok tunnel (printed in console)
=============================================================================
"""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import warnings
import os
import threading

warnings.filterwarnings("ignore")

app = FastAPI(title="Crop Recommendation System")

# CORS for ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# DISTRICT → LAT/LON MAP  (Karnataka districts)
# -----------------------------------------------------------------------------
DISTRICT_COORDS = {
    "Bagalkot"          : (16.1691, 75.6966),
    "Bangalore Rural"   : (13.1986, 77.7066),
    "Bangalore Urban"   : (12.9716, 77.5946),
    "Belgaum"           : (15.8497, 74.4977),
    "Bellary"           : (15.1394, 76.9214),
    "Bidar"             : (17.9104, 77.5199),
    "Bijapur"           : (16.8302, 75.7100),
    "Chamarajanagar"    : (11.9261, 76.9440),
    "Chikballapur"      : (13.4355, 77.7315),
    "Chikmagalur"       : (13.3153, 75.7754),
    "Chitradurga"       : (14.2251, 76.3980),
    "Dakshina Kannada"  : (12.8438, 75.2479),
    "Davangere"         : (14.4644, 75.9218),
    "Dharwad"           : (15.4589, 75.0078),
    "Gadag"             : (15.4166, 75.6322),
    "Gulbarga"          : (17.3297, 76.8343),
    "Hassan"            : (13.0033, 76.1004),
    "Haveri"            : (14.7939, 75.3996),
    "Kodagu"            : (12.4244, 75.7382),
    "Kolar"             : (13.1357, 78.1290),
    "Koppal"            : (15.3548, 76.1551),
    "Mandya"            : (12.5218, 76.8951),
    "Mysore"            : (12.2958, 76.6394),
    "Raichur"           : (16.2120, 77.3439),
    "Ramanagara"        : (12.7157, 77.2793),
    "Shimoga"           : (13.9299, 75.5681),
    "Tumkur"            : (13.3379, 77.1173),
    "Udupi"             : (13.3409, 74.7421),
    "Uttara Kannada"    : (14.7860, 74.6808),
    "Yadgir"            : (16.7714, 77.1384),
}

# -----------------------------------------------------------------------------
# LOAD MODELS  (graceful fallback if files missing — for dev/demo)
# -----------------------------------------------------------------------------
try:
    model1     = joblib.load("model1_npk.pkl")
    label_enc1 = joblib.load("model1_label_encoder.pkl")
    MODEL1_LOADED = True
    print("✅ Model 1 (NPK) loaded")
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
    print("✅ Model 2 (Regional CSV) loaded")
except Exception as e:
    print(f"⚠️  Could not load model2 CSV: {e}")
    model2_df = pd.DataFrame()
    DISTRICTS = list(DISTRICT_COORDS.keys())
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
    """Returns districts, seasons, and their lat/lon coordinates."""
    coords = {d: DISTRICT_COORDS.get(d, (15.3173, 75.7139)) for d in DISTRICTS}
    return {
        "districts"        : DISTRICTS,
        "seasons"          : SEASONS,
        "district_coords"  : coords,
    }

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
        coords = DISTRICT_COORDS.get(district, (15.3173, 75.7139))
        return {
            "status"  : "ok",
            "results" : result,
            "district": district,
            "season"  : season,
            "coords"  : {"lat": coords[0], "lon": coords[1]},
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/browse")
async def api_browse(
    district: str = "Bagalkot",
    season: str = "Kharif",
    top_n: int = 8
):
    if not MODEL2_LOADED:
        return {"status": "ok", "results": [], "coords": DISTRICT_COORDS.get(district, (15.3173, 75.7139))}
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
        coords = DISTRICT_COORDS.get(district, (15.3173, 75.7139))
        return {
            "status" : "ok",
            "results": results,
            "coords" : {"lat": coords[0], "lon": coords[1]},
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/districts/map")
async def get_all_district_map():
    """Returns all districts with their coordinates for map overview."""
    data = []
    for d in DISTRICTS:
        coords = DISTRICT_COORDS.get(d, (15.3173, 75.7139))
        data.append({"district": d, "lat": coords[0], "lon": coords[1]})
    return {"status": "ok", "districts": data}

# -----------------------------------------------------------------------------
# STARTUP: ngrok tunnel
# -----------------------------------------------------------------------------
NGROK_URL = None

def start_ngrok(port: int = 8000):
    global NGROK_URL
    try:
        from pyngrok import ngrok, conf
        # Optional: set auth token via env var NGROK_AUTHTOKEN
        token = os.environ.get("NGROK_AUTHTOKEN")
        if token:
            conf.get_default().auth_token = token

        tunnel    = ngrok.connect(port, "http")
        NGROK_URL = tunnel.public_url
        print(f"\n{'='*60}")
        print(f"  🌐 Public URL (ngrok) : {NGROK_URL}")
        print(f"  📍 Local URL          : http://localhost:{port}")
        print(f"{'='*60}\n")
    except ImportError:
        print("⚠️  pyngrok not installed — skipping tunnel (pip install pyngrok)")
    except Exception as e:
        print(f"⚠️  ngrok tunnel failed: {e}")

@app.get("/api/ngrok-url")
async def get_ngrok_url():
    return {"url": NGROK_URL or f"http://localhost:8000"}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import webbrowser

    PORT = 8000
    LOCAL_URL = f"http://127.0.0.1:{PORT}"

    print("\n🌾 Starting Crop Recommendation System — Karnataka")
    print(f"   Districts loaded : {len(DISTRICTS)}")
    print(f"   Model 1 (NPK)    : {'✅' if MODEL1_LOADED else '⚠️  fallback'}")
    print(f"   Model 2 (Region) : {'✅' if MODEL2_LOADED else '⚠️  fallback'}")
    print(f"\n   ✅ Local URL : {LOCAL_URL}")
    print(f"   Opening browser automatically...\n")

    # Start ngrok in background thread before uvicorn blocks
    threading.Thread(target=start_ngrok, args=(PORT,), daemon=True).start()

    # Auto-open browser after short delay
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(LOCAL_URL)
    threading.Thread(target=open_browser, daemon=True).start()

    # Use 127.0.0.1 instead of 0.0.0.0 to avoid firewall/timeout issues
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=False)