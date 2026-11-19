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

## Training / epochs

The shipped model was **not trained in this repository**. It is a fine-tuned
`google/mobilenet_v2_1.0_224` checkpoint. The model card reports:

| Setting | Value |
| --- | --- |
| Epochs | **6** |
| Batch size | 256 |
| Learning rate | 5e-5 (Adam, linear schedule, 20% warm-up) |
| Dataset | PlantVillage (Kaggle "New Plant Diseases Dataset") |
| Reported eval accuracy | **95.41%** |
| Cross-entropy loss | 0.15 |

The repository's optional script `ai-ml/train_disease_model.py` is a different
pipeline: it is configured for `EPOCHS = 20` with early stopping (patience 4) and
`ReduceLROnPlateau`. It was not used to produce the ONNX file that ships with the
app. If you re-train with that script, 20 is the maximum, not a guarantee.

## Supported crops

The 38 classes cover **14 crops**: Tomato, Potato, Corn (Maize), Apple, Grape,
Peach, Bell Pepper, Cherry, Strawberry, Blueberry, Raspberry, Soybean, Squash
and Orange.

It **cannot** diagnose wheat, rice, cotton, sugarcane, mustard, millet or other
field crops. The API rejects an unsupported `plant_type` instead of returning a
confident disease from a different crop. The frontend dropdown is limited to
the 14 supported crops for the same reason.

## Inference safeguards

- **Test-time augmentation:** probabilities are averaged over the original
  image, a horizontal mirror, and a 5% centre zoom. This makes the result less
  sensitive to leaf orientation and framing.
- **Plant-type filter:** the softmax is restricted to the selected crop when
  that crop has classes in the model. If the full model is very confident about
  a different crop (>=60%) while the selected crop is weak (<20%), the image's
  prediction wins and `crop_mismatch` is set so the UI can warn the user.
- **Photo-quality hints:** `image_quality` reports green-pixel ratio, brightness
  and a simple blur score. These are warnings, not hard rejections.
- **Low-confidence flag:** anything below 45% is marked `low_confidence`.

## Accuracy caveat

PlantVillage images are clean, single-leaf, uniform-background photos. Field
photos with soil, shadows, multiple leaves or phone-camera noise are much harder.
Independent field-photo benchmarks (e.g. PlantDoc) are far lower than the 95%
lab number, so treat this as a demo-grade diagnosis, not an agronomic guarantee.

`class_indices.json` maps each class name to its output index. It was generated
from the `labels.json` published with the ONNX conversion.

Verified on PlantVillage sample images:

| Image | Prediction | Confidence |
| --- | --- | --- |
| Tomato late blight | Tomato: Late Blight | 98.3% |
| Potato early blight | Potato: Early Blight | 99.2% |
| Corn common rust | Corn (Maize): Common Rust | 98.9% |
| Apple scab | Apple Scab | 87.2% |
| Healthy tomato | Healthy Tomato Plant | 99.99% |

Note: the files in `ai-ml/models/` come from the placeholder/training pipeline and
are **not** used by the backend disease endpoint.

## Crop model (for comparison)

`ai-ml/models/crop_model.pkl` is a scikit-learn **RandomForestClassifier**, not a
neural network. It has **no epochs**: it builds 100 decision trees in one `fit()`
call (`n_estimators=100` in `ai-ml/train_model.py`). The dataset has 2,200 rows
and 22 crop classes, and contains no location/season column, which is why the
backend adds a transparent region/season filter (`backend/utils/crop_advisor.py`)
on top of its output.

The 22 classes in the shipped label encoder are:

`apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute,
kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange,
papaya, pigeonpeas, pomegranate, rice, watermelon`.

Sugarcane is **not** one of the crop-model classes. It is still recommended by
the transparent regional rules, and the UI labels it as a rule-based crop.
