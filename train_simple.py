# Legacy script. Uses a random image split that leaks data. Do not use for results.

"""
train_simple.py  —  Fast ECG Classifier (No GPU needed)
=========================================================
Uses image features + SVM instead of deep learning.
Trains in ~2 minutes instead of 2 hours.
Expected accuracy: 70-85%

Run from D:\\ECG_Project:  python train_simple.py
"""

import os, json, pickle
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle

BASE_DIR   = "data/ecg_images"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ecg_model_svm.pkl")
NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE   = (128, 128)
CLASSES    = ["Normal", "Abnormal", "MI"]

os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_features(img_path):
    """
    Extract meaningful features from ECG image:
    - HOG-like gradient features (captures waveform shape)
    - Pixel intensity histogram (captures signal density)
    - Row/column projections (captures lead structure)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # 1. Grayscale + remove red grid
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, (0,30,100), (12,255,255))
    m2  = cv2.inRange(hsv, (155,30,100), (180,255,255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[cv2.bitwise_or(m1,m2) > 0] = 255
    gray = gray.astype(np.float32) / 255.0

    features = []

    # 2. Pixel intensity histogram (32 bins)
    hist, _ = np.histogram(gray, bins=32, range=(0,1))
    features.extend(hist / hist.sum())

    # 3. Horizontal projection (row sums) — captures lead row structure
    row_proj = gray.mean(axis=1)   # 128 values
    features.extend(row_proj)

    # 4. Vertical projection (col sums) — captures R-peak positions
    col_proj = gray.mean(axis=0)   # 128 values
    features.extend(col_proj)

    # 5. Gradient magnitude histogram — captures waveform sharpness
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_hist, _ = np.histogram(mag, bins=32, range=(0,1))
    features.extend(grad_hist / (grad_hist.sum() + 1e-6))

    # 6. Block mean features (4x4 grid = 16 blocks)
    h, w = gray.shape
    bh, bw = h//4, w//4
    for i in range(4):
        for j in range(4):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            features.append(block.mean())
            features.append(block.std())

    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
#  LOAD DATASET
# ══════════════════════════════════════════════════════════════════

def load_split(split):
    X, y = [], []
    split_dir = os.path.join(BASE_DIR, split)
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  ⚠ {split}/{cls} not found — skipping")
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        print(f"  {split}/{cls}: {len(imgs)} images", end="", flush=True)
        ok = 0
        for fname in imgs:
            feat = extract_features(os.path.join(cls_dir, fname))
            if feat is not None:
                X.append(feat)
                y.append(cls_idx)
                ok += 1
        print(f" → {ok} features extracted")
    return np.array(X), np.array(y)


print("\n═══ Loading train set ═══")
X_train, y_train = load_split("train")
print(f"  Total: {len(X_train)} samples, {X_train.shape[1]} features each")

print("\n═══ Loading val set ═══")
X_val, y_val = load_split("val")
print(f"  Total: {len(X_val)} samples")

if len(X_train) == 0:
    print("❌ No training data found"); exit(1)

# Shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)


# ══════════════════════════════════════════════════════════════════
#  TRAIN SVM
# ══════════════════════════════════════════════════════════════════
print("\n═══ Training SVM classifier ═══")

# Class weights to handle imbalance
from collections import Counter
counts = Counter(y_train)
total  = len(y_train)
class_weight = {i: total/(len(CLASSES)*counts[i]) for i in range(len(CLASSES)) if i in counts}
print(f"  Class weights: { {CLASSES[k]:round(v,2) for k,v in class_weight.items()} }")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(
        kernel       = 'rbf',
        C            = 10,
        gamma        = 'scale',
        class_weight = class_weight,
        probability  = True,      # needed for confidence scores
        random_state = 42,
    ))
])

print("  Fitting... (this takes 1-3 minutes)")
pipeline.fit(X_train, y_train)
print("  ✅ Training complete")


# ══════════════════════════════════════════════════════════════════
#  EVALUATE
# ══════════════════════════════════════════════════════════════════
print("\n═══ Evaluation ═══")

y_pred_train = pipeline.predict(X_train)
y_pred_val   = pipeline.predict(X_val)

print(f"  Train accuracy: {accuracy_score(y_train, y_pred_train)*100:.1f}%")
print(f"  Val   accuracy: {accuracy_score(y_val,   y_pred_val  )*100:.1f}%")

print("\nPer-class report (val):")
print(classification_report(y_val, y_pred_val, target_names=CLASSES,
                             zero_division=0))


# ══════════════════════════════════════════════════════════════════
#  SAVE MODEL
# ══════════════════════════════════════════════════════════════════
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

class_names = {str(i): cls for i, cls in enumerate(CLASSES)}
with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\n✅ Model saved  → {MODEL_PATH}")
print(f"✅ Names saved  → {NAMES_PATH}")
print("\nNow update ecg_analysis.py to use SVM model.")
print("Run:  python update_analysis.py")