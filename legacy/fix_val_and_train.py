"""
fix_val_and_train.py  —  Fix val set + retrain
===============================================
Run from D:\\ECG_Project:  python fix_val_and_train.py

Problem: val set has only 8 MI images — model can't evaluate properly.
Fix: move 20% of train images into val for each class (stratified split).
Then retrain with correct evaluation.
"""

import os, json, shutil, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

random.seed(42)
np.random.seed(42)

BASE_DIR   = "data/ecg_images"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ecg_model.h5")
NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE   = (224, 224)
BATCH      = 16
VAL_SPLIT  = 0.20      # 20% of each class goes to val
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ["Normal", "Abnormal", "MI"]

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — Move val images BACK to train (reset val set)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 1: Resetting val → train ═══")
for cls in CLASSES:
    val_dir   = os.path.join(BASE_DIR, "val",   cls)
    train_dir = os.path.join(BASE_DIR, "train", cls)
    if not os.path.exists(val_dir):
        continue
    os.makedirs(train_dir, exist_ok=True)
    imgs = [f for f in os.listdir(val_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    for f in imgs:
        shutil.move(os.path.join(val_dir, f),
                    os.path.join(train_dir, f"val_{f}"))
    print(f"  Moved {len(imgs)} val/{cls} → train/{cls}")

# ══════════════════════════════════════════════════════════════════
#  STEP 2 — Stratified 80/20 split from train → val
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 2: Stratified 80/20 split ═══")
for cls in CLASSES:
    train_dir = os.path.join(BASE_DIR, "train", cls)
    val_dir   = os.path.join(BASE_DIR, "val",   cls)
    if not os.path.exists(train_dir):
        print(f"  ⚠ train/{cls} not found — skipping")
        continue
    os.makedirs(val_dir, exist_ok=True)
    imgs = [f for f in os.listdir(train_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    random.shuffle(imgs)
    n_val = max(10, int(len(imgs) * VAL_SPLIT))   # at least 10 per class
    for f in imgs[:n_val]:
        shutil.move(os.path.join(train_dir, f),
                    os.path.join(val_dir, f))
    print(f"  {cls}: {len(imgs)-n_val} train  /  {n_val} val")

# ══════════════════════════════════════════════════════════════════
#  STEP 3 — Verify distribution
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 3: Final distribution ═══")
valid_classes = []
for split in ["train", "val"]:
    print(f"{split}/")
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, split, cls)
        if not os.path.exists(path):
            continue
        n = len([f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
        print(f"  {cls}: {n}")
        if split == "train" and n >= 20:
            valid_classes.append(cls)

valid_classes = sorted(set(valid_classes))
print(f"\nTraining on: {valid_classes}")
if len(valid_classes) < 2:
    print("❌ Not enough data"); exit(1)

# ══════════════════════════════════════════════════════════════════
#  STEP 4 — Data generators
# ══════════════════════════════════════════════════════════════════
tgen = ImageDataGenerator(
    rescale=1./255, rotation_range=5,
    width_shift_range=0.08, height_shift_range=0.08,
    zoom_range=0.08, brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)
vgen = ImageDataGenerator(rescale=1./255)

td = tgen.flow_from_directory(
    f"{BASE_DIR}/train", target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='categorical',
    shuffle=True, classes=valid_classes
)
vd = vgen.flow_from_directory(
    f"{BASE_DIR}/val", target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='categorical',
    shuffle=False, classes=valid_classes
)

NC = len(td.class_indices)
cnames = {str(v): k for k, v in td.class_indices.items()}
with open(NAMES_PATH, "w") as f:
    json.dump(cnames, f, indent=2)
print(f"\nClass map: {td.class_indices}")
print(f"Train: {td.samples}   Val: {vd.samples}")

cw  = compute_class_weight("balanced",
                            classes=np.unique(td.classes),
                            y=td.classes)
class_weights = dict(enumerate(cw))
print(f"Class weights: { {cnames[str(k)]: round(v,2) for k,v in class_weights.items()} }")

# ══════════════════════════════════════════════════════════════════
#  STEP 5 — Build model
# ══════════════════════════════════════════════════════════════════
print("\n═══ Building model ═══")
bb  = EfficientNetB0(weights="imagenet", include_top=False,
                     input_shape=(*IMG_SIZE, 3))
bb.trainable = False

inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
x   = bb(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(256, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(NC, activation="softmax")(x)
model = Model(inp, out)

cbs = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, save_weights_only=False, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
]

# ══════════════════════════════════════════════════════════════════
#  STEP 6 — Phase 1: frozen
# ══════════════════════════════════════════════════════════════════
print("\n═══ Phase 1: Frozen backbone (15 epochs) ═══")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(td, validation_data=vd, epochs=15,
          class_weight=class_weights, callbacks=cbs, verbose=1)

# ══════════════════════════════════════════════════════════════════
#  STEP 7 — Phase 2: fine-tune
# ══════════════════════════════════════════════════════════════════
print("\n═══ Phase 2: Fine-tuning (25 epochs) ═══")
bb.trainable = True
for layer in bb.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(td, validation_data=vd, epochs=25,
          class_weight=class_weights, callbacks=cbs, verbose=1)

# ══════════════════════════════════════════════════════════════════
#  STEP 8 — Evaluate
# ══════════════════════════════════════════════════════════════════
print("\n═══ Final Evaluation ═══")
model = tf.keras.models.load_model(MODEL_PATH)
loss, acc = model.evaluate(vd, verbose=0)
print(f"\nVal accuracy: {acc*100:.2f}%")

vd.reset()
y_pred  = np.argmax(model.predict(vd, verbose=0), axis=1)
y_true  = vd.classes
labels  = [cnames[str(i)] for i in range(NC)]

print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=labels))
print(f"\n✅ Model  → {MODEL_PATH}")
print(f"✅ Names  → {NAMES_PATH}")
print("\nRestart Flask — CNN will auto-load.")