"""
ecg_analysis.py – FURTHER HARDENED VERSION (Reduced False AFib)
"""

import os, json, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter, detrend

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ecg_model_svm.pkl")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "model", "class_names.json")

_model = None
_class_names = None


# ================= VISUAL =================
def visualize_signal(sig, peaks, bpm=0, fs=125):
    try:
        plt.figure(figsize=(12, 5))
        t = np.arange(len(sig)) / fs
        plt.plot(t, sig, color="#00e5a0", linewidth=1.2)
        if len(peaks) > 0:
            plt.scatter(t[peaks], sig[peaks], color='red', s=25, zorder=5)
        plt.title(f"Extracted ECG Signal - Heart Rate: {bpm} bpm")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Normalized Amplitude")
        plt.grid(True, alpha=0.3)
        os.makedirs("static", exist_ok=True)
        plt.savefig("static/plot.png", dpi=200, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


# ================= MODEL (Still Disabled) =================
def _load_model():
    global _model, _class_names
    if _model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(NAMES_PATH) as f:
            _class_names = json.load(f)
        return True
    except Exception:
        return False


def _predict_with_model(image_path):
    return None  # Disabled


# ================= SIGNAL EXTRACTION =================
def _row_signal(gray):
    h, w = gray.shape
    top = int(h * 0.08)
    bot = int(h * 0.92)
    roi = gray[top:bot, :].astype(np.float32)

    sig = np.zeros(w)
    for x in range(w):
        col = roi[:, x]
        min_val = np.min(col)
        if min_val < 180:                     # Dark trace threshold
            sig[x] = np.argmin(col)
        else:
            sig[x] = (bot - top) / 2.0

    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-8)
    return -sig


# ================= PREPROCESSING =================
def _preprocess(sig, fs=125):
    sig = detrend(sig)

    # Baseline wander removal
    b, a = butter(2, 0.5 / (fs / 2), btype='high')
    sig = filtfilt(b, a, sig)

    # ECG bandpass
    b, a = butter(4, [0.5, 40], btype='band', fs=fs)
    sig = filtfilt(b, a, sig)

    try:
        sig = savgol_filter(sig, 11, 3)
    except Exception:
        pass
    return sig


# ================= PEAK DETECTION =================
def _detect_peaks(sig, fs=125):
    sig_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)

    height_thresh = np.percentile(sig_norm, 70)
    min_distance = max(int(fs * 0.28), 20)   # ~200 bpm max

    peaks, props = find_peaks(
        sig_norm,
        height=height_thresh,
        distance=min_distance,
        prominence=0.18,           # Stricter
        width=(None, int(fs * 0.12))
    )

    if len(peaks) > 0:
        strong_mask = props['peak_heights'] > np.median(props['peak_heights']) * 0.75
        peaks = peaks[strong_mask]

    return peaks


# ================= METRICS =================
def _compute_metrics(peaks, fs=125):
    if len(peaks) < 4:
        return dict(bpm=0, rmssd=0.0, rr_std_ms=0.0, cv=0.0, irregular=False,
                    confidence="low", rr_intervals=[])

    rr = np.diff(peaks) / fs * 1000.0   # ms

    # Strong outlier rejection
    if len(rr) > 3:
        median_rr = np.median(rr)
        # Remove RR that deviate >30% from median or are physiologically impossible
        valid = (rr > 300) & (rr < 1800) & (np.abs(rr - median_rr) < 0.30 * median_rr)
        rr_clean = rr[valid]
    else:
        rr_clean = rr

    if len(rr_clean) < 3:
        return dict(bpm=0, rmssd=0.0, rr_std_ms=0.0, cv=0.0, irregular=False,
                    confidence="low", rr_intervals=rr.tolist())

    mr = np.mean(rr_clean)
    bpm = int(round(60000 / mr)) if mr > 0 else 0

    rmssd = np.sqrt(np.mean(np.diff(rr_clean) ** 2)) if len(rr_clean) > 1 else 0.0
    std = np.std(rr_clean)
    cv = std / mr if mr > 0 else 0.0

    # Irregular only if variability is convincingly high
    irregular = (cv > 0.22) and (rmssd > 80) and (len(rr_clean) >= 6)

    if len(rr_clean) >= 12:
        conf = "high"
    elif len(rr_clean) >= 7:
        conf = "medium"
    else:
        conf = "low"

    return dict(
        bpm=bpm,
        rmssd=round(rmssd, 1),
        rr_std_ms=round(std, 1),
        cv=round(cv, 3),
        irregular=irregular,
        confidence=conf,
        rr_intervals=[round(x, 1) for x in rr_clean]
    )


# ================= CLASSIFICATION =================
def _rule_classify(m, quality, num_peaks):
    bpm = m["bpm"]
    rmssd = m["rmssd"]
    cv = m["cv"]
    irreg = m["irregular"]
    conf = m["confidence"]

    if quality == "poor" or conf == "low" or num_peaks < 6:
        return dict(
            condition="Uncertain / Possible Artifact",
            risk="Moderate",
            description=f"Signal quality insufficient for reliable diagnosis. Estimated HR {bpm} bpm."
        )

    if 50 <= bpm <= 100 and not irreg and rmssd < 90:
        return dict(
            condition="Normal Sinus Rhythm",
            risk="Low",
            description=f"Regular rhythm at {bpm} bpm."
        )

    # Much stricter AFib rule to avoid false positives like in your report
    if irreg and rmssd > 150 and cv > 0.25 and num_peaks >= 10:
        return dict(
            condition="Suspected Atrial Fibrillation (AFib)",
            risk="High",
            description=f"Highly irregular rhythm (RMSSD={rmssd} ms, CV={cv}). Medical review recommended."
        )

    if bpm < 50:
        return dict(
            condition="Bradycardia",
            risk="Moderate",
            description=f"Low heart rate ({bpm} bpm)."
        )

    if bpm > 100:
        return dict(
            condition="Tachycardia",
            risk="Moderate",
            description=f"High heart rate ({bpm} bpm)."
        )

    if irreg:
        return dict(
            condition="Irregular Rhythm (Possible Artifact)",
            risk="Moderate",
            description=f"Irregular pattern detected ({bpm} bpm, RMSSD={rmssd} ms). May be noise."
        )

    return dict(
        condition="Normal Sinus Rhythm",
        risk="Low",
        description=f"Stable rhythm at {bpm} bpm."
    )


# ================= MAIN =================
def analyze_ecg(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image not found.")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sig = _row_signal(gray)
        fs = 125.0

        sig = _preprocess(sig, fs)
        peaks = _detect_peaks(sig, fs)

        duration_sec = len(sig) / fs
        quality = "good" if len(peaks) >= 8 else "poor"

        m = _compute_metrics(peaks, fs)
        res = _rule_classify(m, quality, len(peaks))

        visualize_signal(sig, peaks, m["bpm"], fs)

        return {
            "bpm": m["bpm"],
            "peaks": len(peaks),
            "rmssd": m["rmssd"],
            "rr_std_ms": m["rr_std_ms"],
            "cv": m["cv"],
            "condition": res["condition"],
            "risk": res["risk"],
            "description": res["description"],
            "confidence": m["confidence"],
            "signal_quality": quality,
            "duration_seconds": round(duration_sec, 1),
            "rr_intervals_ms": m["rr_intervals"],
            "source": "rule_based",
            "disclaimer": "Screening tool only. Not for medical diagnosis. High RMSSD often indicates noise/missed beats on paper ECGs."
        }

    except Exception as e:
        return {
            "bpm": 0,
            "peaks": 0,
            "condition": "Analysis Error",
            "risk": "Unknown",
            "description": f"Processing failed: {str(e)}",
            "confidence": "low",
            "signal_quality": "poor",
            "disclaimer": "Please try a clearer photo without grid interference."
        }