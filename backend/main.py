import os
import sys

# Windows consoles default to cp1252, which cannot encode the status emojis.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Load environment variables from backend/.env before other modules read them
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import requests

from routers import auth, marketplace
from utils.disease_detector import detector

app = FastAPI(
    title="Smart Farmer Assistant API",
    description="AI-powered agriculture API for crop recommendations and disease detection",
    version="1.0.0",
)

app.include_router(auth.router)
app.include_router(marketplace.router)

# ─── CORS ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── City Coordinates ───
CITY_COORDS = {
    "Cairo": (30.0444, 31.2357),
    "Mansoura": (31.0409, 31.3785),
    "Alexandria": (31.2001, 29.9187),
    "Tanta": (30.7865, 31.0004),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Paris": (48.8566, 2.3522),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8688, 151.2093),
    "Lagos": (6.5244, 3.3792),
    "Nairobi": (-1.2921, 36.8219),
    "Karachi": (24.8607, 67.0011),
    "Dhaka": (23.8103, 90.4125),
    "Colombo": (6.9271, 79.8612),
    "Bangkok": (13.7563, 100.5018),
}

# ─── ML Models ───
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai-ml", "models")

crop_model = None
scaler = None
label_encoder = None

try:
    with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb") as f:
        crop_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ AI Models loaded successfully")
except Exception as e:
    print(f"⚠️  Models not loaded (running in fallback mode): {e}")


def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch live weather plus a 30-day rainfall total from Open-Meteo (no API key)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m"
            f"&daily=precipitation_sum&past_days=30&forecast_days=1&timezone=auto"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            payload = resp.json()
            current = payload["current"]
            daily = payload.get("daily", {})
            rain_values = [v for v in daily.get("precipitation_sum", []) if v is not None]
            rain_30d = round(float(sum(rain_values)), 1)

            # The crop model was trained on rainfall values between ~20 and
            # ~300 mm. Clamp the observed 30-day total into that range so the
            # model is not fed out-of-distribution values during dry spells.
            rainfall = max(20.0, min(300.0, rain_30d))

            return {
                "temp": current["temperature_2m"],
                "humidity": current["relative_humidity_2m"],
                "rainfall": rainfall,
                "rain_30d": rain_30d,
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    return {"temp": 25.0, "humidity": 70.0, "rainfall": 100.0, "rain_30d": 100.0}


def geocode_city(city: str) -> tuple | None:
    """Geocode a city name to lat/lon using Open-Meteo geocoding API."""
    # First try our local dict
    for known_city, coords in CITY_COORDS.items():
        if city.lower() in known_city.lower() or known_city.lower() in city.lower():
            return coords

    # Fall back to geocoding API
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city)}&count=1"
        resp = requests.get(geo_url, timeout=8)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return results[0]["latitude"], results[0]["longitude"]
    except Exception as e:
        print(f"Geocoding error: {e}")

    return None


# ─── Routes ───

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Smart Farmer Assistant API",
        "version": "1.0.0",
        "models_loaded": crop_model is not None,
        "endpoints": ["/", "/health", "/recommend", "/disease-predict"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": crop_model is not None,
    }


@app.post("/recommend")
async def recommend(data: dict):
    city = str(data.get("city", "")).strip()
    if not city:
        raise HTTPException(status_code=400, detail="Please provide a city name.")

    coords = geocode_city(city)
    if coords is None:
        raise HTTPException(status_code=404, detail=f"Could not find the city '{city}'.")
    lat, lon = coords
    weather = fetch_weather(lat, lon)

    def _number(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    soil = {key: _number(data.get(key)) for key in ("N", "P", "K", "ph")}
    has_soil = all(value is not None for value in soil.values())

    # The soil model is only meaningful when real soil-test values are supplied.
    # With no soil data we use the transparent weather rules below instead of
    # feeding made-up defaults to the model (which produced biased results).
    if crop_model and scaler and label_encoder and has_soil:
        try:
            features = [
                soil["N"],
                soil["P"],
                soil["K"],
                weather["temp"],
                weather["humidity"],
                soil["ph"],
                weather["rainfall"],
            ]
            features_arr = np.array([features])
            features_scaled = scaler.transform(features_arr)
            prediction = crop_model.predict(features_scaled)
            crop_name = str(label_encoder.inverse_transform(prediction)[0]).title()
            probs = crop_model.predict_proba(features_scaled)[0]
            top_indices = np.argsort(probs)[-3:][::-1]
            top_crops = [str(c).title() for c in label_encoder.inverse_transform(top_indices)]
            confidence = round(float(probs[top_indices[0]]) * 100, 2)

            return {
                "recommended_crop": crop_name,
                "confidence": confidence,
                "alternatives": top_crops[1:],
                "method": "ml_model",
                "weather_used": weather,
                "tips": [
                    f"Live weather for {city}: {weather['temp']}°C, {weather['humidity']}% humidity, "
                    f"~{weather['rain_30d']} mm rain in the last 30 days.",
                    f"The AI model recommends {crop_name} with {confidence}% confidence "
                    f"using soil values N={soil['N']:g}, P={soil['P']:g}, K={soil['K']:g}, pH={soil['ph']:g}.",
                    "Always confirm with a soil test and local extension advice before planting.",
                ],
            }
        except Exception as e:
            print(f"ML prediction error: {e}")

    # Transparent weather-based fallback (no soil data supplied)
    temp = weather["temp"]
    humidity = weather["humidity"]
    rainfall = weather.get("rain_30d", weather["rainfall"])

    if temp < 18:
        crop, confidence, alternatives = "Wheat", 88.0, ["Mustard", "Peas"]
    elif rainfall >= 150 or (temp > 30 and humidity > 70):
        crop, confidence, alternatives = "Rice", 86.5, ["Sugarcane", "Jute"]
    elif humidity < 45:
        crop, confidence, alternatives = "Millet", 82.0, ["Sorghum", "Mustard"]
    elif temp > 25 and humidity < 68:
        crop, confidence, alternatives = "Cotton", 83.2, ["Sorghum", "Maize"]
    else:
        crop, confidence, alternatives = "Maize", 84.7, ["Sorghum", "Millet"]

    tips = [
        f"Live weather for {city}: {weather['temp']}°C, {weather['humidity']}% humidity, "
        f"~{weather['rain_30d']} mm rain in the last 30 days.",
        f"{crop} is recommended based on current temperature and humidity.",
        "Add your soil test values (N, P, K, pH) to get a soil-based AI recommendation.",
    ]

    return {
        "recommended_crop": crop,
        "confidence": confidence,
        "alternatives": alternatives,
        "method": "weather_rules",
        "weather_used": weather,
        "tips": tips,
    }


@app.post("/disease-predict")
async def disease_predict(image: UploadFile = File(...), plant_type: str = Form(...)):
    """Diagnose plant disease from leaf image using Deep Learning."""

    # Validate file type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Read image bytes
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty image file received.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    # Use the Deep Learning model!
    result = detector.predict(contents, plant_type)
    
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
