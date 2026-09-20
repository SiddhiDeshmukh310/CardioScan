import tensorflow as tf
import cv2
import numpy as np
import os
import random

model = tf.keras.models.load_model(
    r"D:\ECG_Project\model\ecg_model.h5"
)

CLASS_NAMES = {
    0: "Abnormal",
    1: "MI",
    2: "Normal"
}

BASE = r"D:\ECG_Project\data\ecg_images\train"

for cls in ["Normal", "Abnormal", "MI"]:

    folder = os.path.join(BASE, cls)

    print(f"\n===== {cls} =====")

    samples = random.sample(os.listdir(folder), 3)

    for file in samples:

        path = os.path.join(folder, file)

        img = cv2.imread(path)
        img = cv2.resize(img, (224,224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, 0)

        pred = model.predict(img, verbose=0)[0]

        idx = np.argmax(pred)

        print(
            file,
            "->",
            CLASS_NAMES[idx],
            pred
        )