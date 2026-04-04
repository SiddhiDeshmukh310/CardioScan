"""
train_model.py  –  ECG Condition Classifier
============================================
Fixes over original:
  • 3-class softmax output  (was binary — wrong for 3 conditions)
  • EfficientNetB0 pretrained backbone  (was random-weight CNN)
  • Two-phase training: freeze backbone → unfreeze top layers (fine-tune)
  • Data augmentation to prevent overfitting on medical images
  • Class-weight balancing for imbalanced datasets
  • Saves best model via ModelCheckpoint (not just last epoch)
  • Exports class_names.json so app.py knows label → condition name

Expected folder structure (created by convert_dataset.py):
    data/ecg_images/train/class0/  ← condition 0 images
    data/ecg_images/train/class1/  ← condition 1 images
    data/ecg_images/train/class2/  ← condition 2 images
    data/ecg_images/val/class0/
    data/ecg_images/val/class1/
    data/ecg_images/val/class2/

Output:
    model/ecg_model.h5        ← best checkpoint
    model/class_names.json    ← {0: "class0", 1: "class1", ...}
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from sklearn.utils.class_weight import compute_class_weight

# ── CONFIG ────────────────────────────────────────────────────────
TRAIN_DIR   = "data/ecg_images/train"
VAL_DIR     = "data/ecg_images/val"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "ecg_model.h5")
NAMES_PATH  = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16          # lower = safer for medical images (small dataset)
EPOCHS_FROZEN  = 10       # phase 1: backbone frozen
EPOCHS_FINETUNE= 20       # phase 2: top layers unfrozen
LR_FROZEN   = 1e-3
LR_FINETUNE = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# ── DATA GENERATORS ───────────────────────────────────────────────
# Augmentation only on train — helps generalise on limited ECG data
train_gen = ImageDataGenerator(
    rescale           = 1.0 / 255,
    rotation_range    = 3,        # ECGs shouldn't rotate much
    width_shift_range = 0.05,
    height_shift_range= 0.05,
    zoom_range        = 0.05,
    horizontal_flip   = False,    # flipping ECG changes clinical meaning
    fill_mode         = 'nearest',
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',   # ← was 'binary', wrong for 3 classes
    shuffle     = True,
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    shuffle     = False,
)

NUM_CLASSES = len(train_data.class_indices)
print(f"\n✅ Classes found: {train_data.class_indices}")
print(f"   Train samples : {train_data.samples}")
print(f"   Val   samples : {val_data.samples}")
print(f"   Num classes   : {NUM_CLASSES}\n")

# ── SAVE CLASS NAMES ──────────────────────────────────────────────
# Maps index → folder name so app.py can decode predictions
class_names = {str(v): k for k, v in train_data.class_indices.items()}
with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"✅ Saved class names → {NAMES_PATH}")

# ── CLASS WEIGHTS (handles imbalanced datasets) ───────────────────
labels_flat = train_data.classes
cw = compute_class_weight("balanced",
                           classes=np.unique(labels_flat),
                           y=labels_flat)
class_weights = dict(enumerate(cw))
print(f"✅ Class weights: {class_weights}\n")

# ── MODEL ─────────────────────────────────────────────────────────
# EfficientNetB0 pretrained on ImageNet — strong feature extractor
# even for medical images (edges, textures, shapes transfer well)

backbone = EfficientNetB0(
    weights     = "imagenet",
    include_top = False,
    input_shape = (*IMG_SIZE, 3),
)
backbone.trainable = False   # freeze for phase 1

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = backbone(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)
model.summary()

# ── CALLBACKS ─────────────────────────────────────────────────────
ckpt = ModelCheckpoint(
    MODEL_PATH, monitor="val_accuracy",
    save_best_only=True, verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=7,
    restore_best_weights=True, verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5,
    patience=3, min_lr=1e-6, verbose=1,
)

# ── PHASE 1: FROZEN BACKBONE ──────────────────────────────────────
print("\n═══ Phase 1: Training head (backbone frozen) ═══")
model.compile(
    optimizer = tf.keras.optimizers.Adam(LR_FROZEN),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

history1 = model.fit(
    train_data,
    validation_data = val_data,
    epochs          = EPOCHS_FROZEN,
    class_weight    = class_weights,
    callbacks       = [ckpt, reduce_lr],
    verbose         = 1,
)

# ── PHASE 2: FINE-TUNE TOP LAYERS ─────────────────────────────────
print("\n═══ Phase 2: Fine-tuning top backbone layers ═══")

# Unfreeze last 30 layers of backbone
backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(LR_FINETUNE),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

history2 = model.fit(
    train_data,
    validation_data = val_data,
    epochs          = EPOCHS_FINETUNE,
    class_weight    = class_weights,
    callbacks       = [ckpt, early_stop, reduce_lr],
    verbose         = 1,
)

# ── FINAL EVALUATION ──────────────────────────────────────────────
print("\n═══ Final Evaluation on Validation Set ═══")
model.load_weights(MODEL_PATH)   # reload best checkpoint
loss, acc = model.evaluate(val_data, verbose=1)
print(f"\n✅ Best val accuracy : {acc*100:.2f}%")
print(f"✅ Model saved       → {MODEL_PATH}")
print(f"✅ Class names saved → {NAMES_PATH}")