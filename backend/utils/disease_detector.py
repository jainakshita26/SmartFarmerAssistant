import os
import json
import io
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageOps

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.onnx")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# Results below this confidence are flagged instead of presented as a diagnosis.
LOW_CONFIDENCE_THRESHOLD = 45.0

# Crops represented in the 38-class PlantVillage model.  The model sometimes
# predicts a disease from a different crop than the one the farmer selected;
# we use these names to (a) restrict the softmax to the selected crop and
# (b) warn when the image looks like another crop.
CROP_NAMES = [
    "Corn (Maize)",
    "Bell Pepper",
    "Apple",
    "Blueberry",
    "Cherry",
    "Grape",
    "Orange",
    "Peach",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]

PLANT_TYPE_ALIASES = {
    "corn": "Corn (Maize)",
    "maize": "Corn (Maize)",
    "corn (maize)": "Corn (Maize)",
    "pepper": "Bell Pepper",
    "bell pepper": "Bell Pepper",
    "chilli": "Bell Pepper",
    "chili": "Bell Pepper",
}


def _crop_of(class_name: str) -> str:
    """Return the crop part of a PlantVillage class name."""
    for crop in CROP_NAMES:
        if crop.lower() in class_name.lower():
            return crop
    # Fallback: "Healthy X Plant", "X with Y".
    if " with " in class_name:
        return class_name.split(" with ", 1)[0].strip()
    return class_name.replace("Healthy ", "").replace(" Plant", "").strip()


def _normalize_plant_type(plant_type: str | None) -> str | None:
    if not plant_type:
        return None
    key = " ".join(plant_type.strip().lower().split())
    if key in PLANT_TYPE_ALIASES:
        return PLANT_TYPE_ALIASES[key]
    for crop in CROP_NAMES:
        if key == crop.lower():
            return crop
    return None


def _quality_check(image: "Image.Image") -> dict:
    """Cheap, non-blocking image-quality hints for the UI."""
    small = image.resize((128, 128))
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_ratio = float((((g > r + 8) & (g > b + 8)) & (g > 40)).mean())
    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    blur_score = float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))

    warnings = []
    if green_ratio < 0.04:
        warnings.append("little green colour detected")
    if brightness < 25:
        warnings.append("image is very dark")
    if blur_score < 15:
        warnings.append("image may be blurry")

    return {
        "green_ratio": round(green_ratio, 3),
        "brightness": round(brightness, 1),
        "blur_score": round(blur_score, 1),
        "warnings": warnings,
    }


# Generic, keyword-based treatment advice. The model's classes are human-readable
# names such as "Tomato with Late Blight" or "Healthy Apple", so matching on the
# condition keyword keeps this in sync with the model without a huge lookup table.
def _advice_for(class_name: str) -> dict:
    n = class_name.lower()

    if "healthy" in n:
        return {
            "treatment": "No treatment necessary — your crop looks healthy.",
            "prevention": "Keep up good practices: balanced fertiliser, proper spacing and regular scouting.",
        }
    if "virus" in n or "mosaic" in n or "curl" in n or "greening" in n:
        return {
            "treatment": "Viral diseases cannot be cured. Remove and destroy infected plants to stop the spread.",
            "prevention": "Use certified disease-free seed/seedlings and control whiteflies, aphids and other vectors.",
        }
    if "mildew" in n:
        return {
            "treatment": "Apply sulphur, potassium bicarbonate or a recommended fungicide; remove badly affected leaves.",
            "prevention": "Improve airflow with wider spacing, avoid overhead watering and keep foliage dry.",
        }
    if "rust" in n:
        return {
            "treatment": "Apply a triazole fungicide at the first sign of pustules and repeat as directed.",
            "prevention": "Plant resistant hybrids, rotate crops and remove volunteer plants that carry the fungus.",
        }
    if "blight" in n:
        return {
            "treatment": "Apply a copper-based or chlorothalonil/mancozeb fungicide every 7–10 days; remove infected leaves.",
            "prevention": "Rotate crops, avoid overhead watering, stake plants for airflow and use certified seed.",
        }
    if "scab" in n:
        return {
            "treatment": "Apply captan or myclobutanil during the susceptible period; remove infected fruit and leaves.",
            "prevention": "Rake up fallen leaves, prune for airflow and choose scab-resistant varieties.",
        }
    if "bacterial" in n or "spot" in n or "septoria" in n or "cercospora" in n:
        return {
            "treatment": "Apply a copper-based bactericide/fungicide, remove affected leaves and avoid working with wet plants.",
            "prevention": "Rotate crops, use clean seed, mulch to stop soil splash and water at the base.",
        }
    if "rot" in n or "esca" in n or "measles" in n:
        return {
            "treatment": "Prune out infected wood/fruit, disinfect tools and apply an appropriate fungicide.",
            "prevention": "Avoid fruit/foliage wounds, keep vines well ventilated and remove crop debris after harvest.",
        }
    if "scorch" in n:
        return {
            "treatment": "Remove scorched leaves and apply a recommended fungicide if the spread continues.",
            "prevention": "Keep plants well watered, mulch the soil and avoid overcrowding.",
        }
    if "mite" in n:
        return {
            "treatment": "Spray insecticidal soap or neem oil, covering the undersides of leaves; repeat after 5–7 days.",
            "prevention": "Check leaf undersides regularly, keep plants hydrated and encourage natural predators.",
        }
    return {
        "treatment": "Consult a local agricultural extension officer for a targeted treatment plan.",
        "prevention": "Ensure proper spacing, crop rotation, balanced nutrition and avoid overhead watering.",
    }


def _display_name(class_name: str) -> str:
    """'Tomato with Late Blight' -> 'Tomato: Late Blight'."""
    if " with " in class_name:
        crop, condition = class_name.split(" with ", 1)
        return f"{crop}: {condition}"
    return class_name


class DiseaseDetector:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.class_indices = {}
        self.index_to_class = {}
        self.crop_indices = {}
        self.load_artifacts()

    def load_artifacts(self):
        if not ONNX_AVAILABLE:
            print("⚠️ ONNX Runtime not installed. DiseaseDetector disabled.")
            return

        if os.path.exists(DISEASE_MODEL_PATH):
            try:
                self.session = ort.InferenceSession(DISEASE_MODEL_PATH)
                self.input_name = self.session.get_inputs()[0].name
                print("✅ ONNX Disease model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Error loading ONNX disease model: {e}")

        if os.path.exists(CLASS_INDICES_PATH):
            try:
                with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                    self.class_indices = json.load(f)
                # Accept either {class_name: index} or {index: class_name}
                if isinstance(list(self.class_indices.values())[0], int):
                    self.index_to_class = {v: k for k, v in self.class_indices.items()}
                else:
                    self.index_to_class = {int(v): k for k, v in self.class_indices.items()}

                # Group output indices by crop so we can restrict a prediction
                # to the plant type the farmer selected.
                self.crop_indices = {}
                for idx, name in self.index_to_class.items():
                    self.crop_indices.setdefault(_crop_of(name), []).append(idx)
            except Exception as e:
                print(f"⚠️ Error loading class indices: {e}")

    def _infer(self, image: "Image.Image") -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return self.session.run(None, {self.input_name: arr})[0][0]

    def _predict_probs(self, image: "Image.Image") -> np.ndarray:
        """Average the original, mirrored and slightly zoomed views.

        Test-time augmentation makes a single photo less sensitive to leaf
        orientation and framing.  The graph already contains the softmax, so
        averaging the returned probability vectors is valid.
        """
        probs = self._infer(image)
        probs = probs + self._infer(ImageOps.mirror(image))

        w, h = image.size
        zoom = image.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
        zoom = zoom.resize((224, 224), Image.LANCZOS)
        probs = probs + self._infer(zoom)
        return probs / 3.0

    def predict(self, image_bytes: bytes, plant_type: str | None = None) -> dict:
        if not self.session or not self.index_to_class:
            return {
                "disease": "System Initialization Error",
                "confidence": 0.0,
                "treatment": "Model not loaded.",
                "prevention": "Please ensure the .onnx model and class_indices.json are present in backend/models/.",
            }

        try:
            # The model needs a square centre crop resized to 224x224, fed as
            # NCHW float32 RGB in the 0-255 range (normalisation is in the graph).
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            quality = _quality_check(image)
            selected_crop = _normalize_plant_type(plant_type)

            # The 38-class model only knows these crops.  Returning a confident
            # disease from a different crop (e.g. wheat -> corn rust) would be
            # misleading, so reject unsupported selections explicitly.
            if plant_type and not selected_crop:
                return {
                    "disease": "Unsupported plant type",
                    "confidence": 0.0,
                    "treatment": "This model can diagnose: " + ", ".join(CROP_NAMES) + ".",
                    "prevention": "Choose one of these crops, or use a model trained on the crop you uploaded.",
                    "alternatives": [],
                    "low_confidence": True,
                    "unsupported_plant_type": True,
                    "plant_type": plant_type,
                    "selected_crop": None,
                    "detected_crop": None,
                    "crop_mismatch": False,
                    "image_quality": quality,
                }

            model_image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            predictions = self._predict_probs(model_image)

            global_idx = int(np.argmax(predictions))
            global_class = self.index_to_class.get(global_idx, "Unknown")
            global_crop = _crop_of(global_class)

            # Restrict the softmax to the selected plant type when possible.
            # This prevents a tomato leaf from being labelled "Potato: Early
            # Blight" just because those classes are visually similar.  If the
            # full model is very confident about a different crop, trust the
            # image and flag the mismatch instead of forcing a weak result.
            best_idx = global_idx
            if selected_crop and self.crop_indices.get(selected_crop):
                candidate_ids = self.crop_indices[selected_crop]
                selected_idx = max(candidate_ids, key=lambda i: predictions[i])
                if (
                    global_crop == selected_crop
                    or predictions[global_idx] < 0.60
                    or predictions[selected_idx] >= 0.20
                ):
                    best_idx = selected_idx

            disease_class = self.index_to_class.get(best_idx, "Unknown")
            best = {
                "disease": _display_name(disease_class),
                "confidence": round(float(predictions[best_idx]) * 100, 2),
            }

            other_ids = [
                int(i)
                for i in np.argsort(predictions)[::-1]
                if int(i) != best_idx
            ][:2]
            top = [best] + [
                {
                    "disease": _display_name(self.index_to_class.get(i, "Unknown")),
                    "confidence": round(float(predictions[i]) * 100, 2),
                }
                for i in other_ids
            ]

            info = _advice_for(disease_class)
            low_confidence = best["confidence"] < LOW_CONFIDENCE_THRESHOLD
            best_crop = _crop_of(disease_class)
            crop_mismatch = bool(selected_crop and best_crop != selected_crop)

            if low_confidence:
                info["treatment"] = (
                    f"Low-confidence result ({best['confidence']}%). "
                    "Try a sharp, well-lit close-up of a single leaf. " + info["treatment"]
                )
            if crop_mismatch:
                info["prevention"] = (
                    f"Note: the image looks more like {best_crop} than {selected_crop}. "
                    "If you selected the wrong plant type, run the scan again with the correct crop. "
                    + info["prevention"]
                )
            if quality["warnings"]:
                info["prevention"] += " Photo quality: " + ", ".join(quality["warnings"]) + "."

            return {
                "disease": best["disease"],
                "confidence": best["confidence"],
                "treatment": info["treatment"],
                "prevention": info["prevention"],
                "alternatives": top[1:],
                "low_confidence": low_confidence,
                "raw_class": disease_class,
                "plant_type": plant_type,
                "selected_crop": selected_crop,
                "detected_crop": global_crop,
                "crop_mismatch": crop_mismatch,
                "unsupported_plant_type": False,
                "image_quality": quality,
            }
        except Exception as e:
            print(f"Prediction Error: {e}")
            return {
                "disease": "Analysis Failed",
                "confidence": 0.0,
                "treatment": f"Error parsing image: {str(e)}",
                "prevention": "Try uploading a clearer photo.",
            }


# Singleton instance
detector = DiseaseDetector()
