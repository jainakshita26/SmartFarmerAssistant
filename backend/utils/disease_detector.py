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
            except Exception as e:
                print(f"⚠️ Error loading class indices: {e}")

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
            image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            img_array = np.asarray(image, dtype=np.float32)
            img_array = np.transpose(img_array, (2, 0, 1))[None, ...]

            predictions = self.session.run(None, {self.input_name: img_array})[0][0]

            top_indices = np.argsort(predictions)[-3:][::-1]
            top = [
                {
                    "disease": _display_name(self.index_to_class.get(int(i), "Unknown")),
                    "confidence": round(float(predictions[i]) * 100, 2),
                }
                for i in top_indices
            ]

            best = top[0]
            disease_class = self.index_to_class.get(int(top_indices[0]), "Unknown")
            info = _advice_for(disease_class)
            low_confidence = best["confidence"] < LOW_CONFIDENCE_THRESHOLD

            if low_confidence:
                info["treatment"] = (
                    f"Low-confidence result ({best['confidence']}%). "
                    "Try a sharp, well-lit close-up of a single leaf. " + info["treatment"]
                )

            return {
                "disease": best["disease"],
                "confidence": best["confidence"],
                "treatment": info["treatment"],
                "prevention": info["prevention"],
                "alternatives": top[1:],
                "low_confidence": low_confidence,
                "raw_class": disease_class,
                "plant_type": plant_type,
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
