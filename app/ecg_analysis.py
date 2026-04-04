"""
ecg_analysis.py  –  ECG Image Analyser
========================================
Primary : SVM model  (model/ecg_model_svm.pkl)
Fallback : Rule-based signal analysis
"""

import os, json, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ecg_model_svm.pkl")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "model", "class_names.json")

_model       = None
_class_names = None


# ══════════════════════════════════════════════════════════════════
#  VISUALISER
# ══════════════════════════════════════════════════════════════════

def visualize_signal(sig, peaks):
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(sig, label="ECG Signal", color="#00e5a0")
        if len(peaks) > 0:
            plt.scatter(peaks, sig[peaks], color='red', zorder=5, label="R-peaks")
        plt.title("ECG Signal with R-peaks")
        plt.legend()
        os.makedirs("static", exist_ok=True)
        plt.savefig("static/plot.png")
        plt.close()
    except Exception as e:
        print(f"[visualize] {e}")


# ══════════════════════════════════════════════════════════════════
#  SVM MODEL
# ══════════════════════════════════════════════════════════════════

def _extract_features(image_path):
    """Extract image features matching train_simple.py."""
    IMG_SIZE = (128, 128)
    img = cv2.imread(image_path)
    if img is None:
        return None
    img  = cv2.resize(img, IMG_SIZE)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, (0, 30, 100), (12, 255, 255))
    m2   = cv2.inRange(hsv, (155, 30, 100), (180, 255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[cv2.bitwise_or(m1, m2) > 0] = 255
    gray = gray.astype(np.float32) / 255.0

    features = []
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    features.extend(hist / hist.sum())
    features.extend(gray.mean(axis=1))
    features.extend(gray.mean(axis=0))
    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    gh, _ = np.histogram(mag, bins=32, range=(0, 1))
    features.extend(gh / (gh.sum() + 1e-6))
    h, w  = gray.shape
    bh, bw = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            features.append(block.mean())
            features.append(block.std())
    return np.array(features, dtype=np.float32)


def _load_model():
    global _model, _class_names
    if _model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        if os.path.exists(NAMES_PATH):
            with open(NAMES_PATH) as f:
                _class_names = json.load(f)
        else:
            _class_names = {"0": "Normal", "1": "Abnormal", "2": "MI"}
        return True
    except Exception as e:
        print(f"[ecg_analysis] Model load failed: {e}")
        return False


def _predict_with_model(image_path: str):
    if not _load_model():
        return None
    try:
        feat = _extract_features(image_path)
        if feat is None:
            return None

        feat2d   = feat.reshape(1, -1)
        probs    = _model.predict_proba(feat2d)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        class_folder = _class_names.get(str(pred_idx), f"class{pred_idx}")

        condition_map = {
            "Normal":   "Normal Sinus Rhythm",
            "Abnormal": "Abnormal / Arrhythmia",
            "MI":       "Myocardial Infarction (MI)",
        }
        risk_map = {
            "Normal":   "Low",
            "Abnormal": "High",
            "MI":       "High",
        }

        condition = condition_map.get(class_folder, class_folder)
        risk      = risk_map.get(class_folder, "Unknown")

        all_probs = {
            condition_map.get(_class_names.get(str(i), f"class{i}"),
                              _class_names.get(str(i), f"class{i}")):
            round(float(probs[i]) * 100, 1)
            for i in range(len(probs))
        }

        return {
            "condition":  condition,
            "risk":       risk,
            "confidence": round(confidence * 100, 1),
            "all_probs":  all_probs,
            "source":     "svm_model",
        }
    except Exception as e:
        print(f"[ecg_analysis] SVM prediction failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
#  RULE-BASED FALLBACK
# ══════════════════════════════════════════════════════════════════

TARGET_W   = 1400
DEFAULT_FS = 150.0
MIN_ROW_H  = 30
LEADS_12x1 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LEADS_3x4  = [["I","aVL","V1","V4"],["II","aVR","V2","V5"],["III","aVF","V3","V6"]]


def _load_and_clean(image_path):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    h, w  = bgr.shape[:2]
    new_h = int(h * TARGET_W / w)
    bgr   = cv2.resize(bgr, (TARGET_W, new_h))
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1    = cv2.inRange(hsv, (0, 30, 100), (12, 255, 255))
    m2    = cv2.inRange(hsv, (155, 30, 100), (180, 255, 255))
    pink  = cv2.inRange(hsv, (0, 5, 190), (25, 60, 255))
    mask  = cv2.bitwise_or(cv2.bitwise_or(m1, m2), pink)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray[mask > 0] = 255
    kh    = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    kv    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    bg    = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_OPEN, kh),
                       cv2.morphologyEx(gray, cv2.MORPH_OPEN, kv))
    trace = cv2.subtract(bg, gray)
    if np.percentile(trace, 95) < 15:
        trace = cv2.subtract(gray, bg)
    return bgr, cv2.GaussianBlur(trace, (3, 3), 0)


def _detect_rows(trace):
    energy = trace.mean(axis=1).astype(np.float32)
    smooth = np.convolve(energy, np.ones(20)/20, mode='same')
    active = (smooth > smooth.mean() * 0.6).astype(np.uint8)
    bands, in_b, s = [], False, 0
    for i, v in enumerate(active):
        if v and not in_b:
            in_b, s = True, i
        elif not v and in_b:
            in_b = False
            if i - s >= MIN_ROW_H:
                bands.append((s, i))
    if in_b and len(active) - s >= MIN_ROW_H:
        bands.append((s, len(active)))
    return bands


def _classify_layout(bands, h, w):
    if not bands:
        return "unknown", bands
    avg_h = np.mean([b[1]-b[0] for b in bands])
    return ("3x4" if len(bands) <= 5 and avg_h > h*0.12 else "12x1"), bands


def _row_signal(row):
    h, w = row.shape
    top  = max(1, int(h*.10))
    bot  = min(h-1, int(h*.90))
    roi  = row[top:bot, :].astype(np.float64)
    sig  = np.zeros(w)
    for x in range(w):
        col = roi[:, x]; s = col.sum()
        sig[x] = (np.arange(len(col))*col).sum()/s if s > 3 else (bot-top)/2
    return -sig


def _bandpass(sig, fs):
    nyq = fs/2
    b, a = butter(3, [max(.5/nyq,1e-4), min(40/nyq,.999)], btype='band')
    return filtfilt(b, a, sig)


def _preprocess(sig, fs):
    sig = sig - np.mean(sig)
    if len(sig) > 40:
        try: sig = _bandpass(sig, fs)
        except: pass
    wl = max(5, min(int(fs*.08)|1, 15))
    if wl%2==0: wl+=1
    try:
        sig = savgol_filter(sig, wl, 3)
    except: pass
    r = sig.max()-sig.min()
    if r > 1e-6: sig = (sig-sig.min())/r
    return sig


def _detect_peaks(sig, fs):
    e = uniform_filter1d(np.diff(sig, prepend=sig[0])**2,
                         size=max(1, int(fs*.06)))
    pks, _ = find_peaks(e, height=max(np.mean(e)+np.std(e),1e-9),
                        distance=max(1, int(.33*fs)))
    hw = max(1, int(fs*.05))
    refined = []
    for p in pks:
        lo = max(0, p-hw); hi = min(len(sig)-1, p+hw)
        refined.append(lo+int(np.argmax(sig[lo:hi+1])))
    return np.unique(refined).astype(int)


def _estimate_fs(bgr):
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bold = cv2.inRange(hsv, (0,50,50), (20,255,210))
    cs   = bold.sum(axis=0).astype(np.float32)
    if cs.max() == 0: return DEFAULT_FS
    cs /= cs.max()
    pks, _ = find_peaks(cs, distance=8, height=0.25)
    if len(pks) < 4: return DEFAULT_FS
    sp = float(np.median(np.diff(pks)))
    return float(np.clip(sp/.2, 40, 500)) if sp >= 4 else DEFAULT_FS


def _compute_metrics(peaks, fs):
    if len(peaks) < 2:
        return dict(bpm=0, rmssd=0., rr_std_ms=0., cv=0.,
                    irregular=False, confidence="low")
    rr    = np.diff(peaks)/fs*1000
    valid = rr[(rr>=280)&(rr<=2100)]
    if len(valid) < 2: valid = rr
    mr    = float(np.mean(valid))
    bpm   = int(np.clip(round(60000/mr), 0, 300)) if mr > 0 else 0
    rmssd = float(np.sqrt(np.mean(np.diff(valid)**2))) if len(valid)>1 else 0.
    std   = float(np.std(valid))
    cv    = std/mr if mr else 0.
    irreg = bool(cv > 0.15 and rmssd > 50)
    conf  = "high" if len(peaks)>=6 else ("medium" if len(peaks)>=3 else "low")
    return dict(bpm=bpm, rmssd=round(rmssd,1), rr_std_ms=round(std,1),
                cv=round(cv,3), irregular=irreg, confidence=conf)


def _rule_classify(m, quality):
    bpm, rmssd, cv, irreg, conf = (
        m["bpm"], m["rmssd"], m["cv"], m["irregular"], m["confidence"])
    if quality == "poor" or conf == "low":
        return dict(condition="Unreadable / Low Quality", risk="Unknown",
                    description="Signal quality too poor. Try a clearer image.")
    if bpm == 0:
        return dict(condition="No ECG Detected", risk="Unknown",
                    description="No R-peaks found.")
    if irreg and rmssd > 120 and cv > 0.20 and 40 <= bpm <= 180 and conf == "high":
        return dict(condition="Atrial Fibrillation (AFib)", risk="High",
                    description=f"Irregular RR (RMSSD={rmssd:.0f} ms, CV={cv:.0%}), {bpm} bpm.")
    if bpm < 60:
        return dict(condition="Bradycardia",
                    risk="High" if bpm < 40 else "Moderate",
                    description=f"Heart rate {bpm} bpm — below normal.")
    if bpm > 100:
        return dict(condition="Tachycardia",
                    risk="High" if bpm > 150 else "Moderate",
                    description=f"Heart rate {bpm} bpm — above normal.")
    if irreg:
        return dict(condition="Irregular Rhythm", risk="Moderate",
                    description=f"Mild irregularity (RMSSD={rmssd:.0f} ms), {bpm} bpm.")
    return dict(condition="Normal Sinus Rhythm", risk="Low",
                description=f"Regular rhythm at {bpm} bpm.")


def _rule_based(image_path: str) -> dict:
    try:
        bgr, trace = _load_and_clean(image_path)
    except Exception as e:
        return _empty_result("Invalid Image", str(e))
    try:
        fs     = _estimate_fs(bgr)
        bands  = _detect_rows(trace)
        layout, bands = _classify_layout(bands, trace.shape[0], trace.shape[1])
        signals = {}
        if not bands:
            signals["II"] = _preprocess(_row_signal(trace), fs)
            layout = "single"
        elif layout == "12x1":
            for i, (y0,y1) in enumerate(bands):
                if i >= len(LEADS_12x1): break
                signals[LEADS_12x1[i]] = _preprocess(_row_signal(trace[y0:y1,:]), fs)
        elif layout == "3x4":
            cw = trace.shape[1]//4
            for r,(y0,y1) in enumerate(bands[:3]):
                for c in range(4):
                    signals[LEADS_3x4[r][c]] = _preprocess(
                        _row_signal(trace[y0:y1, c*cw:(c+1)*cw]), fs)
            if len(bands) >= 4:
                y0,y1 = bands[3]
                signals["rhythm_II"] = _preprocess(_row_signal(trace[y0:y1,:]), fs)
        else:
            signals["II"] = _preprocess(_row_signal(trace), fs)

        sig, lead_used = None, "II"
        for lead in ["rhythm_II","II","V5","I"]:
            if lead in signals:
                sig, lead_used = signals[lead], lead
                break
        if sig is None and signals:
            lead_used = list(signals.keys())[0]
            sig = signals[lead_used]
        if sig is None or len(sig) < 10:
            return _empty_result("Weak Signal", "Signal too weak.")

        dr         = sig.max()-sig.min()
        flat_ratio = np.sum(np.abs(np.diff(sig)) < 0.001)/len(sig)
        quality    = "poor" if dr < 0.25 or flat_ratio > 0.6 else "good"
        pks        = _detect_peaks(sig, fs)
        if len(pks) < 3:   quality = "poor"
        elif len(pks) < 6: quality = "moderate"

        m   = _compute_metrics(pks, fs)
        res = _rule_classify(m, quality)
        visualize_signal(sig, pks)

        return {
            "bpm":             m["bpm"],
            "peaks":           len(pks),
            "rmssd":           m["rmssd"],
            "rr_std_ms":       m["rr_std_ms"],
            "cv":              m["cv"],
            "condition":       res["condition"],
            "risk":            res["risk"],
            "description":     res["description"],
            "confidence":      m["confidence"],
            "signal_quality":  quality,
            "fs_estimated":    round(fs, 2),
            "layout_detected": layout,
            "lead_used":       lead_used,
            "leads_found":     list(signals.keys()),
            "all_probs":       {},
            "cnn_confidence":  None,
            "rule_condition":  res["condition"],
            "rule_risk":       res["risk"],
            "source":          "rule_based",
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return _empty_result("Analysis Error", str(e))


def _empty_result(condition, description):
    return {
        "bpm": 0, "peaks": 0, "rmssd": 0., "rr_std_ms": 0., "cv": 0.,
        "condition": condition, "risk": "Unknown", "description": description,
        "confidence": "low", "signal_quality": "poor",
        "fs_estimated": 0., "layout_detected": "unknown",
        "lead_used": "none", "leads_found": [],
        "all_probs": {}, "cnn_confidence": None,
        "rule_condition": condition, "rule_risk": "Unknown",
        "source": "rule_based",
    }


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def analyze_ecg(image_path: str) -> dict:
    rule = _rule_based(image_path)
    svm  = _predict_with_model(image_path)

    if svm:
        return {
            "bpm":             rule["bpm"],
            "peaks":           rule["peaks"],
            "rmssd":           rule["rmssd"],
            "rr_std_ms":       rule["rr_std_ms"],
            "cv":              rule["cv"],
            "signal_quality":  rule["signal_quality"],
            "fs_estimated":    rule.get("fs_estimated", 0),
            "layout_detected": rule.get("layout_detected", "unknown"),
            "lead_used":       rule.get("lead_used", "unknown"),
            "leads_found":     rule.get("leads_found", []),
            "condition":       svm["condition"],
            "risk":            svm["risk"],
            "description":     f"SVM model ({svm['confidence']}% confidence). Signal: {rule['description']}",
            "confidence":      rule["confidence"],
            "source":          "svm_model",
            "cnn_confidence":  svm["confidence"],
            "all_probs":       svm["all_probs"],
            "rule_condition":  rule["condition"],
            "rule_risk":       rule["risk"],
        }

    rule["description"] = "⚠ No model found — signal analysis only. Run train_simple.py first. | " + rule["description"]
    return rule


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ecg_sample.jpg"
    import json as _j
    print(_j.dumps(analyze_ecg(path), indent=2))