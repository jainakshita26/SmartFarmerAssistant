"""Region- and season-aware crop advisory layer.

The RandomForest in ``ai-ml/train_model.py`` was trained on the Kaggle
Crop_recommendation dataset.  That dataset only contains N, P, K, temperature,
humidity, pH and rainfall - it has **no location or season columns**.  As a
result the model can happily recommend cotton for Kashmir, because cotton and
the Kashmir sample may have similar soil/weather numbers in the training data.

This module is a transparent agronomic rule layer on top of the model:

* map the geocoded state / elevation to an agro-climatic region
* work out the current cropping season (Kharif / Rabi / Zaid)
* keep only crops that are agronomically plausible for that region+season
* re-rank the model probabilities inside that shortlist

It is deliberately *not* another trained model.  When the model has no
plausible candidate, the caller falls back to explicit region rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

# Dataset labels (lower-case, as returned by LabelEncoder) -> display names.
MODEL_DISPLAY = {
    "rice": "Rice",
    "maize": "Maize",
    "chickpea": "Chickpea",
    "kidneybeans": "Kidney Beans",
    "pigeonpeas": "Pigeon Peas",
    "mothbeans": "Moth Beans",
    "mungbean": "Mung Bean",
    "blackgram": "Black Gram",
    "muskmelon": "Muskmelon",
    "lentil": "Lentil",
    "pomegranate": "Pomegranate",
    "banana": "Banana",
    "mango": "Mango",
    "grapes": "Grapes",
    "watermelon": "Watermelon",
    "apple": "Apple",
    "orange": "Orange",
    "papaya": "Papaya",
    "coconut": "Coconut",
    "cotton": "Cotton",
    "jute": "Jute",
    "coffee": "Coffee",
}

DISPLAY_MODEL = {v: k for k, v in MODEL_DISPLAY.items()}

# Human-readable growing season for the 22 crops the RandomForest was trained
# on.  These are display hints only - the model itself has no season feature.
MODEL_CROP_SEASONS = {
    "Rice": "Kharif",
    "Maize": "Kharif / Rabi",
    "Chickpea": "Rabi",
    "Kidney Beans": "Kharif",
    "Pigeon Peas": "Kharif",
    "Moth Beans": "Kharif",
    "Mung Bean": "Kharif / Zaid",
    "Black Gram": "Kharif",
    "Muskmelon": "Zaid / Summer",
    "Lentil": "Rabi",
    "Pomegranate": "Year-round",
    "Banana": "Year-round",
    "Mango": "Summer",
    "Grapes": "Year-round",
    "Watermelon": "Zaid / Summer",
    "Apple": "Temperate",
    "Orange": "Winter",
    "Papaya": "Year-round",
    "Coconut": "Year-round",
    "Cotton": "Kharif",
    "Jute": "Kharif",
    "Coffee": "Year-round",
}


def model_crop_details() -> list[dict]:
    """The actual 22 display classes of the crop model, with season hints."""
    return [
        {
            "name": name,
            "key": key,
            "season": MODEL_CROP_SEASONS.get(name, "Seasonal"),
        }
        for key, name in MODEL_DISPLAY.items()
    ]

# Indian states / UTs -> broad agro-climatic region.  Open-Meteo returns
# ``admin1`` values such as "Jammu and Kashmir" or "Tamil Nadu".
STATE_REGION = {
    "jammu and kashmir": "himalayan",
    "jammu & kashmir": "himalayan",
    "ladakh": "himalayan",
    "himachal pradesh": "himalayan",
    "uttarakhand": "himalayan",
    "sikkim": "himalayan",
    "arunachal pradesh": "himalayan",
    "nagaland": "himalayan",
    "manipur": "himalayan",
    "mizoram": "himalayan",
    "meghalaya": "himalayan",
    "tripura": "himalayan",
    "punjab": "north_west",
    "haryana": "north_west",
    "chandigarh": "north_west",
    "delhi": "north_west",
    "nct of delhi": "north_west",
    "uttar pradesh": "north_west",
    "rajasthan": "west",
    "gujarat": "west",
    "maharashtra": "west",
    "goa": "west",
    "madhya pradesh": "central",
    "chhattisgarh": "central",
    "bihar": "east",
    "jharkhand": "east",
    "west bengal": "east",
    "odisha": "east",
    "orissa": "east",
    "assam": "east",
    "tamil nadu": "south",
    "kerala": "south",
    "karnataka": "south",
    "andhra pradesh": "south",
    "telangana": "south",
    "puducherry": "south",
}

# Crops that make agronomic sense in each region/season.  The lists include
# common crops that are not in the 22-class dataset (wheat, mustard, ...) so
# they can be used by the rule-based fallback.
REGION_PROFILES = {
    "himalayan": {
        "kharif": ["Maize", "Rice", "Apple", "Kidney Beans", "Pigeon Peas", "Lentil"],
        "rabi": ["Wheat", "Barley", "Mustard", "Peas", "Apple", "Lentil", "Chickpea", "Potato"],
        "zaid": ["Maize", "Watermelon", "Muskmelon", "Vegetables"],
        "avoid": ["Cotton", "Sugarcane", "Jute", "Mango", "Banana", "Coconut", "Coffee"],
    },
    "north_west": {
        "kharif": ["Rice", "Maize", "Cotton", "Sugarcane", "Millet", "Pigeon Peas",
                   "Moth Beans", "Mung Bean", "Okra", "Watermelon"],
        "rabi": ["Wheat", "Barley", "Mustard", "Peas", "Lentil", "Chickpea",
                 "Potato", "Maize"],
        "zaid": ["Watermelon", "Muskmelon", "Mung Bean", "Maize", "Vegetables"],
        "avoid": ["Coconut", "Coffee"],
    },
    "west": {
        "kharif": ["Cotton", "Rice", "Maize", "Millet", "Sorghum", "Pigeon Peas",
                   "Moth Beans", "Mung Bean", "Black Gram", "Sugarcane", "Banana",
                   "Papaya"],
        "rabi": ["Chickpea", "Wheat", "Lentil", "Mustard", "Maize", "Onion",
                 "Grapes", "Pomegranate", "Orange"],
        "zaid": ["Watermelon", "Muskmelon", "Mung Bean", "Maize", "Vegetables"],
        "avoid": ["Apple", "Coffee"],
    },
    "central": {
        "kharif": ["Rice", "Maize", "Cotton", "Sugarcane", "Soybean", "Pigeon Peas",
                   "Mung Bean", "Black Gram", "Sorghum", "Millet", "Papaya"],
        "rabi": ["Wheat", "Chickpea", "Lentil", "Mustard", "Peas", "Maize",
                 "Pomegranate", "Orange"],
        "zaid": ["Watermelon", "Muskmelon", "Mung Bean", "Maize", "Vegetables"],
        "avoid": ["Coconut", "Coffee", "Apple"],
    },
    "south": {
        "kharif": ["Rice", "Maize", "Cotton", "Sugarcane", "Millet", "Sorghum",
                   "Pigeon Peas", "Mung Bean", "Black Gram", "Banana", "Mango",
                   "Coconut", "Coffee", "Watermelon", "Papaya"],
        "rabi": ["Rice", "Maize", "Millet", "Sorghum", "Chickpea", "Lentil",
                 "Banana", "Mango", "Coconut", "Grapes", "Pomegranate", "Orange",
                 "Papaya"],
        "zaid": ["Watermelon", "Muskmelon", "Mung Bean", "Maize", "Vegetables"],
        "avoid": ["Apple"],
    },
    "east": {
        "kharif": ["Rice", "Jute", "Maize", "Sugarcane", "Mung Bean", "Black Gram",
                   "Pigeon Peas", "Banana", "Papaya"],
        "rabi": ["Potato", "Mustard", "Wheat", "Lentil", "Chickpea", "Peas",
                 "Maize", "Watermelon", "Papaya"],
        "zaid": ["Watermelon", "Muskmelon", "Mung Bean", "Maize", "Vegetables"],
        "avoid": ["Apple", "Coffee", "Coconut"],
    },
    "other": {
        "kharif": ["Rice", "Maize", "Cotton", "Sugarcane", "Millet", "Sorghum"],
        "rabi": ["Wheat", "Barley", "Mustard", "Peas", "Lentil", "Chickpea", "Potato"],
        "zaid": ["Watermelon", "Muskmelon", "Mung Bean", "Maize", "Vegetables"],
        "avoid": [],
    },
}

def rule_crop_details() -> list[dict]:
    """Crops the transparent region rules can suggest outside the ML model.

    ``Vegetables`` is a placeholder category in the region profiles, not a
    specific crop, so it is intentionally not returned to the UI.
    """
    model_crops = set(MODEL_DISPLAY.values())
    season_order = ("Kharif", "Rabi", "Zaid")
    found: dict[str, set[str]] = {}
    for profile in REGION_PROFILES.values():
        for season_key, season_label in (
            ("kharif", "Kharif"),
            ("rabi", "Rabi"),
            ("zaid", "Zaid"),
        ):
            for crop in profile.get(season_key, []):
                if crop in model_crops or crop == "Vegetables":
                    continue
                found.setdefault(crop, set()).add(season_label)

    return [
        {
            "name": crop,
            "season": " / ".join(s for s in season_order if s in seasons),
            "source": "region_rules",
        }
        for crop, seasons in sorted(found.items())
    ]


REGION_LABELS = {
    "himalayan": "Himalayan / highland",
    "north_west": "North-West plains",
    "west": "Western India",
    "central": "Central India",
    "south": "Peninsular / South India",
    "east": "Eastern India",
    "other": "General",
}

SEASON_MONTHS = {
    "Kharif": "Jun-Oct",
    "Rabi": "Nov-Mar",
    "Zaid": "Apr-May",
}


def get_season(month: int | None = None) -> str:
    """Indian cropping season for a calendar month."""
    m = month or datetime.now().month
    if m in (6, 7, 8, 9, 10):
        return "Kharif"
    if m in (11, 12, 1, 2, 3):
        return "Rabi"
    return "Zaid"


def region_for(state: str | None, country: str | None, elevation: float | None) -> str:
    """Resolve a geocoded state/elevation to one of the region profiles."""
    if state:
        key = state.strip().lower()
        if key in STATE_REGION:
            return STATE_REGION[key]
        for name, region in STATE_REGION.items():
            if name in key or key in name:
                return region

    # Non-India locations use the generic profile.  Elevation alone (e.g.
    # Denver) should not be treated as a Himalayan farm.
    if country and country.strip().lower() not in ("india", "in"):
        return "other"

    # No usable state: high elevation is a strong highland signal inside India.
    if elevation is not None and elevation >= 1200:
        return "himalayan"

    return "other"


def allowed_model_crops(region: str, season: str) -> list[str]:
    """Display names from the 22-class model that are plausible right now."""
    profile = REGION_PROFILES.get(region, REGION_PROFILES["other"])
    avoid = set(profile.get("avoid", []))
    seasonal = profile.get(season.lower(), [])
    allowed = [c for c in seasonal if c in DISPLAY_MODEL and c not in avoid]

    # If the season list has no overlap with the model, allow any non-avoided
    # model crop rather than returning an empty list.
    if not allowed:
        allowed = [c for c in MODEL_DISPLAY.values() if c not in avoid]
    return allowed


def rerank_ml(
    classes: Iterable[str],
    probabilities: Iterable[float],
    region: str,
    season: str,
    min_probability: float = 0.12,
) -> dict | None:
    """Pick the best model crop that is also region/season suitable.

    ``classes`` are the raw LabelEncoder labels (e.g. ``"cotton"``), while
    ``probabilities`` are the corresponding model probabilities.  Returns
    ``None`` when no suitable crop reaches ``min_probability``; the caller then
    uses the explicit region rules.
    """
    allowed = set(allowed_model_crops(region, season))
    candidates = []
    for raw, prob in zip(classes, probabilities):
        display = MODEL_DISPLAY.get(str(raw).lower())
        if display and display in allowed:
            candidates.append((display, float(prob)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    if not candidates or candidates[0][1] < min_probability:
        return None

    top = candidates[:3]
    return {
        "crop": top[0][0],
        "confidence": round(top[0][1] * 100, 1),
        "confidence_type": "model_probability",
        "alternatives": [c for c, _ in top[1:]],
        "method": "ml_model+region_filter",
    }


def rule_recommendation(
    region: str,
    season: str,
    temperature: float,
    humidity: float,
    rainfall_30d: float,
    elevation: float | None = None,
) -> dict:
    """Transparent fallback when the ML shortlist has no confident crop."""
    profile = REGION_PROFILES.get(region, REGION_PROFILES["other"])
    crops = list(profile.get(season.lower(), []))
    if not crops:
        crops = list(REGION_PROFILES["other"].get(season.lower(), ["Maize"]))

    # Weather nudges, applied only when the preferred crop is valid for the
    # region/season.  They never introduce an off-season crop.
    def promote(crop: str) -> None:
        if crop in crops:
            crops.remove(crop)
            crops.insert(0, crop)

    if temperature < 18:
        if "Wheat" in crops:
            promote("Wheat")
        elif "Barley" in crops:
            promote("Barley")
    elif rainfall_30d >= 150 or (temperature > 30 and humidity > 70):
        promote("Rice")
    elif humidity < 45:
        promote("Millet")

    # In highland areas, apple is a better Kharif choice than a generic cereal
    # only when the weather is not clearly cold-season.
    if region == "himalayan" and season == "Kharif" and (elevation or 0) >= 1200:
        promote("Apple")

    crop = crops[0]
    confidence = 86.0 if crop in ("Wheat", "Rice", "Maize") else 82.0
    return {
        "crop": crop,
        "confidence": confidence,
        "confidence_type": "rule_score",
        "alternatives": crops[1:3],
        "method": "region_rules",
    }


def suitability_note(region: str, season: str, state: str | None) -> str:
    """Short human-readable explanation added to the response tips."""
    where = state or REGION_LABELS.get(region, "your area")
    return (
        f"Region: {where} ({REGION_LABELS.get(region, region)}), "
        f"season: {season} ({SEASON_MONTHS[season]}). "
        "The recommendation respects local crop suitability, not just soil nutrients."
    )
