# Disease model

`disease_model.onnx` is a **real, trained** MobileNetV2 (1.0 / 224) classifier for
38 PlantVillage classes across 14 crops. It replaces the earlier placeholder model
built by `ai-ml/models/generate_placeholder.py` (which was untrained and returned
an almost-uniform ~2.6% for every class).

- Original weights: [`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
- ONNX conversion: [`shahriar-ahmed-seam/leafwise`](https://github.com/shahriar-ahmed-seam/leafwise) (MIT)
- Input: RGB, centre-crop to square, resize to 224x224, float32 **0-255** in NCHW
  layout. Normalisation is already folded into the ONNX graph.
- Output: softmax probabilities over the 38 classes.

`class_indices.json` maps each class name to its output index. It was generated
from the `labels.json` published with the ONNX conversion.

Verified on PlantVillage sample images:

| Image | Prediction | Confidence |
| --- | --- | --- |
| Tomato late blight | Tomato: Late Blight | 98.4% |
| Potato early blight | Potato: Early Blight | 99.6% |
| Corn common rust | Corn (Maize): Common Rust | 98.3% |

Note: the files in `ai-ml/models/` come from the placeholder/training pipeline and
are **not** used by the backend disease endpoint.
