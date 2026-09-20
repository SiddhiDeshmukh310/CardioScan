import os
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

CLASSES = ['MI', 'NORM', 'OTHER_ABNORMAL']

# Load Keras EfficientNet Model
_model = None
def get_model():
    global _model
    if _model is None:
        model_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'model', 'best_efficientnet_leakfree.h5'),
            os.path.join(os.path.dirname(__file__), '..', 'model', 'best_efficientnet.h5')
        ]
        for p in model_paths:
            if os.path.exists(p):
                try:
                    _model = tf.keras.models.load_model(p)
                    print(f'Loaded model from {p}')
                    break
                except Exception as e:
                    print(f'Failed to load model from {p}: {e}')
    return _model

def analyze_ecg(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Unable to read uploaded image.')

    # 1. EfficientNet Model Prediction
    model = get_model()
    if model is not None:
        img_resized = cv2.resize(img, (224, 224))
        img_arr = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_arr, axis=0)
        preds = model.predict(img_batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        prediction_label = CLASSES[top_idx]
        confidence = float(preds[top_idx] * 100)
        all_probs = {CLASSES[i]: float(preds[i] * 100) for i in range(len(CLASSES))}
    else:
        prediction_label = 'UNKNOWN'
        confidence = 0.0
        all_probs = {'MI': 33.3, 'NORM': 33.3, 'OTHER_ABNORMAL': 33.4}

    # 2. Rule-Based OpenCV Signal Analysis (Supplementary)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    signal = np.mean(gray, axis=0)
    signal_inverted = 255.0 - signal
    signal_norm = (signal_inverted - np.min(signal_inverted)) / (np.ptp(signal_inverted) + 1e-6)

    peaks, _ = find_peaks(signal_norm, distance=30, height=0.4)
    estimated_hr = len(peaks) * 6  # Rough estimate

    # Plot signal visualization
    plt.figure(figsize=(10, 3))
    plt.plot(signal_norm, color='#10b981', label='Extracted Signal')
    if len(peaks) > 0:
        plt.scatter(peaks, signal_norm[peaks], color='red', s=20, label='Detected Peaks')
    plt.title('Rule-Based Signal Extraction (OpenCV)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'static', 'plot.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    return {
        'prediction': prediction_label,
        'confidence': round(confidence, 1),
        'all_probs': all_probs,
        'rule_based_analysis': {
            'label': 'rule-based signal analysis',
            'estimated_hr_bpm': estimated_hr,
            'peaks_detected': len(peaks),
            'signal_plot_url': '/static/plot.png'
        }
    }
