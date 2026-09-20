import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    r"D:\ECG_Project\data\ecg_images\val",
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

model = tf.keras.models.load_model("best_efficientnet.h5")

preds = model.predict(val_data)

y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(val_data.class_indices.keys())
    )
)

print("\nConfusion Matrix\n")
print(confusion_matrix(y_true, y_pred))