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
from utils.disease_detector import detector, CROP_NAMES
from utils.crop_advisor import (
    MODEL_DISPLAY,
    MODEL_CROP_SEASONS,
    allowed_model_crops,
    get_season,
    model_crop_details,
    region_for,
    rerank_ml,
    rule_crop_details,
    rule_recommendation,
    suitability_note,
)

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

# ─── City / region metadata ───
# The old version only returned (lat, lon).  We now also keep the state and
# country so the crop adviser can apply regional suitability rules.
CITY_META = {
    "cairo": {"lat": 30.0444, "lon": 31.2357, "state": "", "country": "Egypt", "name": "Cairo"},
    "mansoura": {"lat": 31.0409, "lon": 31.3785, "state": "", "country": "Egypt", "name": "Mansoura"},
    "alexandria": {"lat": 31.2001, "lon": 29.9187, "state": "", "country": "Egypt", "name": "Alexandria"},
    "tanta": {"lat": 30.7865, "lon": 31.0004, "state": "", "country": "Egypt", "name": "Tanta"},
    "delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "country": "India", "name": "Delhi"},
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "country": "India", "name": "Mumbai"},
    "london": {"lat": 51.5074, "lon": -0.1278, "state": "", "country": "United Kingdom", "name": "London"},
    "new york": {"lat": 40.7128, "lon": -74.0060, "state": "New York", "country": "United States", "name": "New York"},
    "paris": {"lat": 48.8566, "lon": 2.3522, "state": "", "country": "France", "name": "Paris"},
    "tokyo": {"lat": 35.6762, "lon": 139.6503, "state": "", "country": "Japan", "name": "Tokyo"},
    "sydney": {"lat": -33.8688, "lon": 151.2093, "state": "New South Wales", "country": "Australia", "name": "Sydney"},
    "lagos": {"lat": 6.5244, "lon": 3.3792, "state": "Lagos", "country": "Nigeria", "name": "Lagos"},
    "nairobi": {"lat": -1.2921, "lon": 36.8219, "state": "Nairobi", "country": "Kenya", "name": "Nairobi"},
    "karachi": {"lat": 24.8607, "lon": 67.0011, "state": "Sindh", "country": "Pakistan", "name": "Karachi"},
    "dhaka": {"lat": 23.8103, "lon": 90.4125, "state": "Dhaka", "country": "Bangladesh", "name": "Dhaka"},
    "colombo": {"lat": 6.9271, "lon": 79.8612, "state": "Western", "country": "Sri Lanka", "name": "Colombo"},
    "bangkok": {"lat": 13.7563, "lon": 100.5018, "state": "Bangkok", "country": "Thailand", "name": "Bangkok"},
}

# Region/state names that the geocoder gets wrong or resolves outside India.
# "Kashmir" is the important one: Open-Meteo returns Kashmir, Iran first, which
# was the reason a Kashmir search could produce cotton.
REGION_ALIASES = {
    "kashmir valley": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "country": "India", "name": "Srinagar (Kashmir Valley)"},
    "jammu and kashmir": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "country": "India", "name": "Srinagar (Jammu and Kashmir)"},
    "jammu & kashmir": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "country": "India", "name": "Srinagar (Jammu and Kashmir)"},
    "j&k": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "country": "India", "name": "Srinagar (Jammu and Kashmir)"},
    "kashmir": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "country": "India", "name": "Srinagar (Kashmir Valley)"},
    "ladakh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh", "country": "India", "name": "Leh (Ladakh)"},
    "himachal": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "country": "India", "name": "Shimla (Himachal Pradesh)"},
    "uttarakhand": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "country": "India", "name": "Dehradun (Uttarakhand)"},
    "punjab": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab", "country": "India", "name": "Ludhiana (Punjab)"},
    "haryana": {"lat": 29.6857, "lon": 76.9905, "state": "Haryana", "country": "India", "name": "Karnal (Haryana)"},
    "rajasthan": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "country": "India", "name": "Jaipur (Rajasthan)"},
    "gujarat": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "country": "India", "name": "Ahmedabad (Gujarat)"},
    "maharashtra": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "country": "India", "name": "Pune (Maharashtra)"},
    "madhya pradesh": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "country": "India", "name": "Bhopal (Madhya Pradesh)"},
    "uttar pradesh": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "country": "India", "name": "Lucknow (Uttar Pradesh)"},
    "bihar": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "country": "India", "name": "Patna (Bihar)"},
    "west bengal": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "country": "India", "name": "Kolkata (West Bengal)"},
    "bengal": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "country": "India", "name": "Kolkata (West Bengal)"},
    "odisha": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha", "country": "India", "name": "Bhubaneswar (Odisha)"},
    "orissa": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha", "country": "India", "name": "Bhubaneswar (Odisha)"},
    "assam": {"lat": 26.1445, "lon": 91.7362, "state": "Assam", "country": "India", "name": "Guwahati (Assam)"},
    "tamil nadu": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "country": "India", "name": "Chennai (Tamil Nadu)"},
    "tamilnadu": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "country": "India", "name": "Chennai (Tamil Nadu)"},
    "kerala": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala", "country": "India", "name": "Thiruvananthapuram (Kerala)"},
    "karnataka": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "country": "India", "name": "Bengaluru (Karnataka)"},
    "andhra pradesh": {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh", "country": "India", "name": "Vijayawada (Andhra Pradesh)"},
    "telangana": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "country": "India", "name": "Hyderabad (Telangana)"},
    "goa": {"lat": 15.4909, "lon": 73.8278, "state": "Goa", "country": "India", "name": "Panaji (Goa)"},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "country": "India", "name": "Bengaluru"},
    "bangalore": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "country": "India", "name": "Bengaluru"},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "country": "India", "name": "Hyderabad"},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "country": "India", "name": "Ahmedabad"},
    "pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "country": "India", "name": "Pune"},
    "jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "country": "India", "name": "Jaipur"},
    "lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "country": "India", "name": "Lucknow"},
    "patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "country": "India", "name": "Patna"},
    "bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "country": "India", "name": "Bhopal"},
    "chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Chandigarh", "country": "India", "name": "Chandigarh"},
    "dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "country": "India", "name": "Dehradun"},
    "shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "country": "India", "name": "Shimla"},
    "leh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh", "country": "India", "name": "Leh"},
    "srinagar": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu and Kashmir", "country": "India", "name": "Srinagar"},
    "guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam", "country": "India", "name": "Guwahati"},
    "bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha", "country": "India", "name": "Bhubaneswar"},
    "kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala", "country": "India", "name": "Kochi"},
    "coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu", "country": "India", "name": "Coimbatore"},
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra", "country": "India", "name": "Nagpur"},
    "indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "country": "India", "name": "Indore"},
    "surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat", "country": "India", "name": "Surat"},
    "ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab", "country": "India", "name": "Ludhiana"},
    "amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab", "country": "India", "name": "Amritsar"},
    "varanasi": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh", "country": "India", "name": "Varanasi"},
    "kanpur": {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh", "country": "India", "name": "Kanpur"},
    "ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "country": "India", "name": "Ranchi"},
    "raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh", "country": "India", "name": "Raipur"},
    "jodhpur": {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan", "country": "India", "name": "Jodhpur"},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "country": "India", "name": "Visakhapatnam"},
    "vijayawada": {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh", "country": "India", "name": "Vijayawada"},
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
    print("âœ… AI Models loaded successfully")
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
                "elevation": payload.get("elevation"),
                "timezone": payload.get("timezone"),
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    return {
        "temp": 25.0,
        "humidity": 70.0,
        "rainfall": 100.0,
        "rain_30d": 100.0,
        "elevation": None,
        "timezone": None,
    }


def geocode_city(city: str) -> dict | None:
    """Resolve a place name to coordinates plus state/country metadata.

    Region/state aliases are checked first because ambiguous names such as
    "Kashmir" otherwise geocode to Kashmir, Iran.  The Open-Meteo result list
    is then preferred for exact-name and India matches.
    """
    key = " ".join(city.strip().lower().split())

    # Longest alias first so "jammu and kashmir" wins over "kashmir".
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        if alias in key:
            return dict(REGION_ALIASES[alias])

    for name, meta in CITY_META.items():
        if name in key or key in name:
            return dict(meta)

    query = city.split(",")[0].strip()
    try:
        params = {"name": query, "count": 10, "language": "en", "format": "json"}
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params=params,
            timeout=8,
        )
        results = resp.json().get("results", []) if resp.status_code == 200 else []

        # If nothing matched, try appending India (the app's main audience).
        if not results:
            params["name"] = f"{query}, India"
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params=params,
                timeout=8,
            )
            results = resp.json().get("results", []) if resp.status_code == 200 else []

        if results:
            indian = [r for r in results if r.get("country", "").lower() == "india"]
            pool = indian or results
            # Prefer the largest place when a name is ambiguous (e.g. there is
            # a small "Bengaluru" village in West Bengal).  Region aliases above
            # already handle Kashmir and other known conflicts.
            chosen = max(pool, key=lambda r: r.get("population") or 0)
            return {
                "lat": chosen["latitude"],
                "lon": chosen["longitude"],
                "state": chosen.get("admin1", ""),
                "country": chosen.get("country", ""),
                "name": chosen.get("name", query),
            }
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
        "endpoints": ["/", "/health", "/model-coverage", "/recommend", "/disease-predict"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": crop_model is not None,
    }


@app.get("/model-coverage")
async def model_coverage():
    """Single source of truth for what each model was actually trained on.

    The frontend uses this instead of hard-coded crop lists, so the UI cannot
    advertise a crop that neither the model nor the regional rules support.
    """
    # Prefer the live LabelEncoder classes: if the shipped model changes, the
    # UI follows automatically.
    if label_encoder is not None:
        crop_classes = []
        for raw in label_encoder.classes_:
            key = str(raw).lower()
            display = MODEL_DISPLAY.get(key, str(raw).title())
            crop_classes.append({
                "name": display,
                "key": key,
                "season": MODEL_CROP_SEASONS.get(display, "Seasonal"),
            })
    else:
        crop_classes = model_crop_details()

    # The disease model's crop list comes from the classes that are actually
    # present in class_indices.json when it is loaded.
    disease_crops = []
    for crop in CROP_NAMES:
        display = crop.replace(" (Maize)", "")
        disease_crops.append({
            "name": display,
            "raw": crop,
            "classes": len(detector.crop_indices.get(crop, [])),
        })

    disease_class_count = len(detector.index_to_class) if detector.index_to_class else 38

    return {
        "crop_model": {
            "name": "RandomForest crop recommendation",
            "count": len(crop_classes),
            "crops": crop_classes,
            "features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            "note": (
                "The soil model is only used when N, P, K and pH are all supplied. "
                "Without soil values the app uses the transparent region/season rules."
            ),
        },
        "crop_rules": {
            "count": len(rule_crop_details()),
            "crops": rule_crop_details(),
            "note": (
                "Transparent region and season rules used when no ML crop is "
                "suitable, or when the best local crop is outside the 22-class model."
            ),
        },
        "disease_model": {
            "name": "MobileNetV2 PlantVillage",
            "crop_count": len(disease_crops),
            "class_count": disease_class_count,
            "crops": disease_crops,
            "note": (
                "Trained on clean, close-up PlantVillage leaf images. "
                "Field photos and unsupported crops are lower confidence or rejected."
            ),
        },
    }


@app.post("/recommend")
async def recommend(data: dict):
    city = str(data.get("city", "")).strip()
    if not city:
        raise HTTPException(status_code=400, detail="Please provide a city name.")

    geo = geocode_city(city)
    if geo is None:
        raise HTTPException(status_code=404, detail=f"Could not find the city '{city}'.")

    weather = fetch_weather(geo["lat"], geo["lon"])
    state = geo.get("state") or ""
    country = geo.get("country") or ""
    elevation = weather.get("elevation")
    season = get_season()
    region = region_for(state, country, elevation)
    suitable = allowed_model_crops(region, season)

    def _number(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    soil = {key: _number(data.get(key)) for key in ("N", "P", "K", "ph")}
    has_soil = all(value is not None for value in soil.values())
    model_top = None

    def _base_tips(crop_name: str, method: str, model_top: str | None = None) -> list[str]:
        tips = [
            f"Live weather for {geo['name']}: {weather['temp']}°C, {weather['humidity']}% humidity, "
            f"~{weather['rain_30d']} mm rain in the last 30 days.",
            suitability_note(region, season, state),
        ]
        if model_top and model_top != crop_name:
            tips.append(
                f"The soil model ranked {model_top} first overall, but it was filtered out "
                f"because it is not suitable for {state or 'this region'} in {season}."
            )
        if method.startswith("ml_model"):
            tips.append(
                f"The AI model's region-filtered choice is {crop_name} "
                f"(N={soil['N']:g}, P={soil['P']:g}, K={soil['K']:g}, pH={soil['ph']:g})."
            )
        else:
            tips.append(
                f"{crop_name} is recommended for {state or 'this region'} in {season} "
                "based on the local crop calendar and current weather."
            )
        tips.append(
            "Always confirm with a soil test and your local agriculture extension officer "
            "before planting."
        )
        return tips

    # The soil model is only meaningful when real soil-test values are supplied.
    # With no soil data we use the transparent region/weather rules below instead
    # of feeding made-up defaults to the model.
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
            probs = crop_model.predict_proba(features_scaled)[0]
            raw_classes = [str(c) for c in label_encoder.classes_]
            model_top_raw = raw_classes[int(np.argmax(probs))]
            model_top = MODEL_DISPLAY.get(model_top_raw.lower(), model_top_raw.title())

            picked = rerank_ml(raw_classes, probs, region, season)
            if picked:
                return {
                    "recommended_crop": picked["crop"],
                    "confidence": picked["confidence"],
                    "confidence_type": picked["confidence_type"],
                    "alternatives": picked["alternatives"],
                    "method": picked["method"],
                    "region": region,
                    "state": state,
                    "season": season,
                    "elevation": elevation,
                    "model_suitable_crops": suitable,
                    "model_top": model_top,
                    "weather_used": weather,
                    "tips": _base_tips(picked["crop"], picked["method"], model_top),
                }

            # The model had no confident region-suitable crop.  Fall through to
            # the region rules rather than showing an implausible prediction.
            print(
                f"ML top '{model_top}' had no suitable candidate for "
                f"{region}/{season}; using region rules."
            )
        except Exception as e:
            print(f"ML prediction error: {e}")

    # Transparent region + season + weather fallback.
    rule = rule_recommendation(
        region,
        season,
        weather["temp"],
        weather["humidity"],
        weather.get("rain_30d", weather["rainfall"]),
        elevation,
    )
    return {
        "recommended_crop": rule["crop"],
        "confidence": rule["confidence"],
        "confidence_type": rule["confidence_type"],
        "alternatives": rule["alternatives"],
        "method": rule["method"],
        "region": region,
        "state": state,
        "season": season,
        "elevation": elevation,
        "model_suitable_crops": suitable,
        "model_top": model_top,
        "weather_used": weather,
        "tips": _base_tips(rule["crop"], rule["method"], model_top),
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
