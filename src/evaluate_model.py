import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    parser = argparse.ArgumentParser(description='Evaluate EfficientNet ECG Model')
    parser.add_argument('--data_dir', type=str, default='data/ecg_images/val', help='Path to validation image directory')
    parser.add_argument('--model_path', type=str, default='model/best_efficientnet.h5', help='Path to trained model file')
    args = parser.parse_args()

    IMG_SIZE = (224, 224)
    val_gen = ImageDataGenerator(rescale=1./255)

    if not os.path.exists(args.data_dir):
        print(f'Validation directory not found at {args.data_dir}. Evaluation skipped.')
        return

    val_data = val_gen.flow_from_directory(
        args.data_dir,
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    if not os.path.exists(args.model_path):
        print(f'Model file not found at {args.model_path}. Evaluation skipped.')
        return

    model = tf.keras.models.load_model(args.model_path)
    preds = model.predict(val_data)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_data.classes

    print('\nClassification Report\n')
    print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

    print('\nConfusion Matrix\n')
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()
