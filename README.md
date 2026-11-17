# Chest X-ray Pneumonia Classification with Machine Learning and Deep Learning

Class mini project (Machine Learning and Deep Learning Techniques in Medicine, Department of Biomedical Engineering, SRM Institute of Science and Technology).

A fine-tuned **MobileNetV2** classifies paediatric chest X-rays as **NORMAL** or **PNEUMONIA**. It is compared with classical ML models (logistic regression, SVM, random forest) trained on frozen deep features, evaluated on an untouched test set, and served in a **Streamlit** app.

> **Educational project. Not a medical device.** Never use it for real diagnosis.

![App screenshot](docs/figures/app_03_pneumonia_bacterial.png)

## Results (official test set, 624 images, threshold 0.5)

| Metric | Value | 95% CI (bootstrap) |
|---|---|---|
| Accuracy | 0.925 | 0.904 – 0.944 |
| Recall (sensitivity) | 0.959 | 0.938 – 0.977 |
| Specificity | 0.868 | 0.824 – 0.910 |
| ROC-AUC | 0.976 | 0.964 – 0.985 |

374 of 390 pneumonia images were detected (16 missed). 31 of 234 normal images were flagged as pneumonia (13.2%).

### Machine learning vs deep learning (test set, threshold 0.5)

| Model | Accuracy | Specificity | ROC-AUC |
|---|---|---|---|
| Logistic regression (frozen features) | 0.808 | 0.509 | 0.944 |
| SVM, RBF (frozen features) | 0.817 | 0.526 | 0.958 |
| Random forest (frozen features) | 0.763 | 0.376 | 0.966 |
| **MobileNetV2 fine-tuned** | **0.925** | **0.868** | **0.976** |

The classical models scored higher on validation (accuracy 0.95–0.97) but did not transfer to the test set. See `docs/figures/fig09_ml_vs_dl_comparison.png` and `fig10_roc_curves_test.png`.

## What makes this project careful

- **Patient-aware split.** One child can have several X-rays, so a random split leaks patients between sets (68% of images in a random 20% validation set came from patients also in training). Patient IDs were taken from file names and the data split with `StratifiedGroupKFold`. The official test set was never used for training decisions.
- **No flips in augmentation.** Horizontal flips mirror the anatomy and the burned-in "R" markers. Our first attempt (flip + reflect fill, `docs/figures/fig05_...`) was rejected; the final model uses ±5° rotation, small shifts, zoom and contrast with constant fill.
- **Two-stage fine-tuning.** Head only first (learning rate 1e-3), then the last 30 base layers (1e-5) with BatchNorm frozen. Class weights handle the 74% / 26% imbalance.
- **Honest evaluation.** Confusion matrices, exact ROC-AUC, bootstrap confidence intervals, and a validation-chosen threshold (0.089) that did **not** transfer to the test set (recall 0.995 but specificity 0.701). Both operating points are reported.
- **Validity checks.** A model that sees only image width, height and aspect ratio reaches test AUC 0.897, so shortcuts may contribute to the results. Patient-ID overlap with the test set and notebook-versus-app prediction consistency were also checked.

## Limitations

- About 13% of healthy test images are flagged as pneumonia.
- Image size and aspect ratio differ between classes, and burned-in markers and text are present. The network may partly use such cues.
- One public dataset of mostly paediatric images from a single source; no external validation.
- The test set has a different class mix (62.5% pneumonia) from the training pool (about 74%).
- The headline threshold 0.5 was adopted after seeing test results (both thresholds are reported).
- One training run per model, default hyper-parameters for the classical baselines, and bootstrap intervals that resample images, not patients.
- Patient IDs for NORMAL images that appear in both pool and test could not be verified.

## Repository structure

```
.
├── app/
│   ├── app.py                 # Streamlit application (final version)
│   ├── requirements.txt
│   ├── final_model.keras      # trained model (21.7 MB, TensorFlow 2.21 / Keras 3.13)
│   └── app_samples/           # optional sample X-rays named NORMAL_*.jpeg / PNEUMONIA_*.jpeg
├── notebooks/
│   └── DL_project.ipynb       # full pipeline with outputs (Google Colab)
├── results/
│   ├── results_table.csv      # validation/test metrics at three thresholds
│   ├── model_comparison.csv   # classical ML vs fine-tuned MobileNetV2
│   ├── stage1_log.csv         # per-epoch training log, head only
│   ├── stage2_log.csv         # per-epoch training log, fine-tuning
│   ├── threshold.json         # validation-chosen (Youden) threshold, 0.0886
│   └── predictions.npz        # saved probabilities and labels (val and test)
├── docs/figures/              # figures used in the report
├── LICENSE
└── README.md
```

## Run the app

Python 3.10–3.13 is required (TensorFlow 2.21 does not support 3.14).

```bash
cd app
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The trained model is included at `app/final_model.keras`. Upload a JPEG/PNG chest X-ray, or pick one of the sample images, and adjust the decision threshold in the sidebar. A lower threshold finds more pneumonia but gives more false alarms.

The app uses exactly the notebook preprocessing (3 channels, resize to 224×224, scaling inside the model). Predictions on test images were verified to match the notebook to four decimals.

## Check the numbers without retraining

`results/predictions.npz` holds the model's predicted probabilities and the true labels for the validation and test sets, so every metric in this README can be recomputed in seconds:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
d = np.load("results/predictions.npz")
y, p = d["y_test"], d["p_test"]
tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int)).ravel()
print(tp, fp, tn, fn, roc_auc_score(y, p))   # 374 31 203 16 0.976...
```

## Reproduce the training

1. Download the dataset from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The archive contains a duplicate inner `chest_xray` folder; use the outer `train`, `val` and `test` folders (5,856 images).
2. Open `notebooks/DL_project.ipynb` in Google Colab and set `DATA_DIR` (dataset folder) and `CKPT_DIR` (where models and figures are saved) in the setup cells. The notebook mounts Google Drive and uses the paths of the original authors, so change them to your own.
3. Run the cells in order. Training takes several minutes per epoch without a GPU.

Note: the last notebook cell writes an earlier draft of `app.py`. The final version is `app/app.py` in this repository.

Environment used: TensorFlow 2.21.0, Keras 3.13.2, scikit-learn, pandas, NumPy, Matplotlib, Pillow, Streamlit.

## Data and credits

- Dataset: Kermany, D. S. et al. (2018). *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell, 172(5), 1122–1131. Kaggle version by P. T. Mooney. Check the licence stated on the dataset page before redistributing images. The dataset itself is not included in this repository.
- MobileNetV2: Sandler, M. et al. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks.* CVPR.
- Built with TensorFlow/Keras, scikit-learn and Streamlit.

## Licence

Code: MIT (see `LICENSE`). The dataset and any sample X-ray images keep their original licence.
