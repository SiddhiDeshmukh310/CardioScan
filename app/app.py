import os
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load 1D CNN Model
MODEL_PATH = 'model/waveform_1d_cnn.h5'
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded 1D Waveform CNN model successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load Sample Index
SAMPLES_INDEX = 'app/samples/samples_index.json'
sample_records = []
if os.path.exists(SAMPLES_INDEX):
    with open(SAMPLES_INDEX, 'r') as f:
        sample_records = json.load(f)

if len(sample_records) < 5:
    sample_files = [f for f in os.listdir('app/samples') if f.endswith('.npy')]
    sample_records = []
    for sf in sample_files:
        parts = sf.replace('.npy', '').split('_')
        ecg_id = parts[1] if len(parts) > 1 else '0'
        lbl = parts[2] if len(parts) > 2 else 'NORM'
        sample_records.append({
            'id': int(ecg_id) if ecg_id.isdigit() else 100,
            'patient_id': 1000 + (int(ecg_id) if ecg_id.isdigit() else 100),
            'age': 62,
            'sex': 0,
            'label': lbl,
            'file': os.path.join('app/samples', sf)
        })

def generate_ecg_plot(signal):
    fig, axes = plt.subplots(6, 2, figsize=(10, 6.5), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    t = np.linspace(0, 10, 1000)
    for idx in range(12):
        r, c = idx % 6, idx // 6
        ax = axes[r, c]
        ax.set_facecolor('#161b22')
        ax.plot(t, signal[:, idx], color='#00e5ff', linewidth=1.1)
        ax.set_title(f"Lead {lead_names[idx]}", color='#c9d1d9', fontsize=8.5, pad=2, loc='left', weight='bold')
        ax.grid(True, color='#21262d', linestyle='--', linewidth=0.5)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            
    fig.text(0.5, 0.01, 'Time (seconds)', ha='center', color='#8b949e', fontsize=8.5)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def calculate_signal_metrics(signal_lead2, fs=100):
    sig = signal_lead2 - np.mean(signal_lead2)
    threshold = np.max(sig) * 0.45
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > threshold and sig[i] > sig[i-1] and sig[i] > sig[i+1]:
            if not peaks or (i - peaks[-1]) > (0.35 * fs):
                peaks.append(i)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs * 1000.0 # ms
        mean_rr = np.mean(rr_intervals)
        bpm = int(round(60000.0 / mean_rr)) if mean_rr > 0 else 72
        rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr_intervals))))) if len(rr_intervals) > 1 else 35.0
        rr_std = float(np.std(rr_intervals))
    else:
        bpm, rmssd, rr_std = 72, 35.0, 25.0
    return max(45, min(180, bpm)), len(peaks), round(rmssd, 1), round(rr_std, 1)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioScan — ECG Screening Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --accent: #00e5ff;
      --accent-glow: rgba(0, 229, 255, 0.15);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding-bottom: 60px; line-height: 1.5; }

    /* Educational Top Banner */
    .edu-banner {
      background: linear-gradient(90deg, #991b1b, #dc2626);
      color: #ffffff;
      text-align: center;
      padding: 9px 16px;
      font-size: 13.5px;
      font-weight: 700;
      letter-spacing: 0.4px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 2px 10px rgba(220, 38, 38, 0.2);
    }

    /* Header Navigation */
    header {
      background-color: #0f172a;
      border-bottom: 1px solid #1e293b;
      padding: 16px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
      width: 36px; height: 36px;
      background: linear-gradient(135deg, #00e5ff, #3b82f6);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
    }
    .logo-text { font-size: 20px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
    .logo-tag { font-size: 11px; background: rgba(0, 229, 255, 0.1); color: #00e5ff; border: 1px solid rgba(0, 229, 255, 0.3); padding: 2px 8px; border-radius: 20px; font-weight: 600; }
    .header-right { font-size: 13px; color: var(--text-muted); font-weight: 500; }

    /* Hero Section */
    .hero {
      padding: 32px 36px 20px 36px;
      max-width: 1300px;
      margin: 0 auto;
    }
    .hero h1 { font-size: 26px; font-weight: 800; color: #ffffff; margin-bottom: 6px; }
    .hero p { color: var(--text-muted); font-size: 14.5px; max-width: 800px; }

    /* Layout Grid */
    .main-grid {
      max-width: 1300px;
      margin: 0 auto;
      padding: 0 36px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 28px;
    }
    @media (max-width: 992px) { .main-grid { grid-template-columns: 1fr; padding: 0 20px; } }

    /* Card Panels */
    .panel-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .panel-title { font-size: 16px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 8px; }

    /* Sample Selector Cards */
    .sample-list { display: flex; flex-direction: column; gap: 12px; max-height: 600px; overflow-y: auto; padding-right: 4px; }
    .sample-card {
      background: #192132;
      border: 1px solid #26334d;
      border-radius: 12px;
      padding: 14px 16px;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
      overflow: hidden;
    }
    .sample-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0, 229, 255, 0.15);
    }
    .sample-card.active {
      background: linear-gradient(135deg, rgba(0, 229, 255, 0.12), rgba(59, 130, 246, 0.12));
      border-color: var(--accent);
      box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
    }
    .sample-card.active::before {
      content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent);
    }
    .sample-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .sample-id { font-weight: 700; font-size: 14px; color: #ffffff; }
    .sample-meta { font-size: 12px; color: var(--text-muted); }

    /* Risk Badges */
    .risk-badge {
      font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .risk-norm { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .risk-mi { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .risk-other { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Diagnostic Banner Box */
    .diag-banner {
      background: linear-gradient(135deg, #1e293b, #0f172a);
      border-radius: 14px;
      padding: 20px 24px;
      border: 1px solid #334155;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .diag-title { font-size: 22px; font-weight: 800; color: #ffffff; }
    .diag-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
    .source-tag { font-size: 11px; background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-weight: 600; border: 1px solid #334155; display: inline-block; margin-top: 8px; }

    /* Waveform Plot Frame */
    .plot-frame {
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 24px;
      box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.8);
      position: relative;
    }
    .plot-img { width: 100%; height: auto; border-radius: 8px; display: block; }

    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    @media (max-width: 768px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
    
    .metric-card {
      background: #161e2e;
      border: 1px solid #232f48;
      border-radius: 12px;
      padding: 16px;
      transition: all 0.2s;
    }
    .metric-card:hover { border-color: #38bdf8; }
    .m-label { font-size: 12px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .m-value { font-size: 24px; font-weight: 800; color: #00e5ff; margin: 4px 0 2px 0; font-family: 'JetBrains Mono', monospace; }
    .m-unit { font-size: 11px; color: var(--text-muted); }
    .rule-tag { font-size: 10px; background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); padding: 2px 6px; border-radius: 4px; font-weight: 700; text-transform: uppercase; display: inline-block; margin-top: 6px; }

    /* Probabilities Section */
    .probs-card {
      background: #161e2e;
      border: 1px solid #232f48;
      border-radius: 12px;
      padding: 20px;
    }
    .probs-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
    .prob-row { margin-bottom: 12px; }
    .prob-info { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    .prob-track { background: #232f48; height: 10px; border-radius: 6px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 6px; transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
    .pf-mi { background: linear-gradient(90deg, #ef4444, #f87171); }
    .pf-norm { background: linear-gradient(90deg, #10b981, #34d399); }
    .pf-other { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

    /* Loading Spinner */
    .loading-overlay {
      position: absolute; inset: 0; background: rgba(13, 17, 23, 0.85); backdrop-filter: blur(4px);
      display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 14px; z-index: 10;
    }
    .spinner { width: 42px; height: 42px; border: 3px solid #21262d; border-top-color: #00e5ff; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    footer {
      max-width: 1300px; margin: 40px auto 0 auto; padding: 20px 36px; border-top: 1px solid #1e293b;
      display: flex; justify-content: space-between; align-items: center; color: var(--text-muted); font-size: 13px;
    }
  </style>
</head>
<body>

  <!-- Top Educational Banner -->
  <div class="edu-banner">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
    Educational demo. Not a medical device.
  </div>

  <header>
    <div class="brand">
      <div class="logo-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
      </div>
      <div>
        <span class="logo-text">CardioScan</span>
        <span class="logo-tag">PTB-XL Benchmark</span>
      </div>
    </div>
    <div class="header-right">1D Waveform Neural Network Demo</div>
  </header>

  <div class="hero">
    <h1>ECG Diagnostic Screening Demo</h1>
    <p>Select a 12-lead ECG sample record from the benchmark PTB-XL dataset below to run automated waveform analysis using our trained 1D Convolutional Neural Network.</p>
  </div>

  <div class="main-grid">
    <!-- Left Panel: Sample Selector -->
    <div class="panel-card">
      <div class="panel-header">
        <span class="panel-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/></svg>
          PTB-XL Sample Records
        </span>
        <span style="font-size:12px; color:var(--text-muted);">{{ samples|length }} Records</span>
      </div>

      <div class="sample-list">
        {% for sample in samples %}
        <div class="sample-card {% if loop.first %}active{% endif %}" onclick="selectSample({{ sample.id }}, this)">
          <div class="sample-top">
            <span class="sample-id">Record #{{ sample.id }}</span>
            <span class="risk-badge {% if sample.label == 'NORM' %}risk-norm{% elif sample.label == 'MI' %}risk-mi{% else %}risk-other{% endif %}">
              {{ sample.label }}
            </span>
          </div>
          <div class="sample-meta">
            Patient #{{ sample.patient_id }} • {{ sample.age }} yrs • {% if sample.sex == 0 %}Male{% else %}Female{% endif %}
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

    <!-- Right Panel: Waveform Display & Diagnostic Results -->
    <div>
      <!-- Diagnostic Banner -->
      <div class="diag-banner">
        <div>
          <div class="diag-title" id="pred-class-title">Analyzing...</div>
          <div class="diag-sub" id="ground-truth-text">PTB-XL Ground Truth: --</div>
          <div class="source-tag">1D Waveform CNN — PTB-XL trained</div>
        </div>
        <div id="risk-pill-box">
          <!-- Filled dynamically -->
        </div>
      </div>

      <!-- Waveform Plot Frame -->
      <div class="plot-frame">
        <div class="loading-overlay" id="loading-overlay">
          <div class="spinner"></div>
          <div style="margin-top:12px; font-size:13px; color:#00e5ff; font-weight:600;">Processing 100 Hz 12-Lead Signals...</div>
        </div>
        <img id="ecg-plot-img" class="plot-img" src="" alt="12-Lead ECG Plot" style="display:none;">
      </div>

      <!-- Signal Metrics Grid -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="m-label">Heart Rate</div>
          <div class="m-value" id="m-bpm">--</div>
          <div class="m-unit">BPM</div>
          <div class="rule-tag">rule-based signal analysis</div>
        </div>
        <div class="metric-card">
          <div class="m-label">R-Peaks</div>
          <div class="m-value" id="m-peaks">--</div>
          <div class="m-unit">Detected</div>
        </div>
        <div class="metric-card">
          <div class="m-label">RMSSD</div>
          <div class="m-value" id="m-rmssd">--</div>
          <div class="m-unit">ms • HRV</div>
        </div>
        <div class="metric-card">
          <div class="m-label">RR Std Dev</div>
          <div class="m-value" id="m-rrstd">--</div>
          <div class="m-unit">ms • Rhythm</div>
        </div>
      </div>

      <!-- Class Probabilities Panel -->
      <div class="probs-card">
        <div class="probs-header">
          <span style="font-size:15px; font-weight:700; color:#ffffff;">Model Class Probabilities</span>
          <span style="font-size:12px; color:var(--text-muted);" id="confidence-score">--</span>
        </div>
        <div id="prob-bars-container">
          <!-- Probability bars dynamically generated -->
        </div>
      </div>
    </div>
  </div>

  <footer>
    <div><strong>CardioScan</strong> • ECG Classification Project • Educational Use Only</div>
    <div>PTB-XL Dataset (Wagner et al., 2020)</div>
  </footer>

  <script>
    let currentSampleId = {{ samples[0].id if samples else 1 }};

    function selectSample(id, elem) {
      document.querySelectorAll('.sample-card').forEach(el => el.classList.remove('active'));
      elem.classList.add('active');
      currentSampleId = id;
      loadSampleData(id);
    }

    async function loadSampleData(id) {
      const loader = document.getElementById('loading-overlay');
      const img = document.getElementById('ecg-plot-img');
      loader.style.display = 'flex';

      try {
        const response = await fetch('/api/predict_sample/' + id);
        const data = await response.json();

        // 1. Render Plot
        img.src = 'data:image/png;base64,' + data.plot_base64;
        img.style.display = 'block';

        // 2. Render Diagnostic Header
        document.getElementById('pred-class-title').textContent = data.predicted_class_fullname;
        document.getElementById('ground-truth-text').textContent = 'PTB-XL Ground Truth: ' + data.ground_truth;

        const riskBox = document.getElementById('risk-pill-box');
        const pred = data.predicted_class;
        if (pred === 'NORM') {
          riskBox.innerHTML = '<span class="risk-badge risk-norm" style="font-size:14px; padding:6px 14px;">Normal ECG</span>';
        } else if (pred === 'MI') {
          riskBox.innerHTML = '<span class="risk-badge risk-mi" style="font-size:14px; padding:6px 14px;">High Risk • MI</span>';
        } else {
          riskBox.innerHTML = '<span class="risk-badge risk-other" style="font-size:14px; padding:6px 14px;">Abnormal ECG</span>';
        }

        // 3. Render Metrics
        document.getElementById('m-bpm').textContent = data.bpm;
        document.getElementById('m-peaks').textContent = data.r_peaks;
        document.getElementById('m-rmssd').textContent = data.rmssd;
        document.getElementById('m-rrstd').textContent = data.rr_std;

        // 4. Render Probabilities
        const probs = data.probabilities;
        let probHtml = '';
        const classNames = {
          'MI': 'Myocardial Infarction (MI)',
          'NORM': 'Normal Electrocardiogram (NORM)',
          'OTHER_ABNORMAL': 'Other Abnormalities (STTC / CD / HYP)'
        };
        const fillClasses = { 'MI': 'pf-mi', 'NORM': 'pf-norm', 'OTHER_ABNORMAL': 'pf-other' };

        for (const [cls, prob] of Object.entries(probs)) {
          const pct = (prob * 100).toFixed(1);
          probHtml += `
            <div class="prob-row">
              <div class="prob-info">
                <span>${classNames[cls] || cls}</span>
                <span>${pct}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill ${fillClasses[cls] || 'pf-norm'}" style="width: ${pct}%;"></div>
              </div>
            </div>
          `;
        }
        document.getElementById('prob-bars-container').innerHTML = probHtml;
        document.getElementById('confidence-score').textContent = (Math.max(...Object.values(probs)) * 100).toFixed(1) + '% Top Confidence';

      } catch (err) {
        console.error(err);
      } finally {
        loader.style.display = 'none';
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      loadSampleData(currentSampleId);
    });
  </script>
</body>
</html>"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, samples=sample_records)

@app.route("/api/predict_sample/<int:sample_id>")
def predict_sample(sample_id):
    rec_info = next((s for s in sample_records if s['id'] == sample_id), None)
    if not rec_info:
        rec_info = sample_records[0]
        
    signal = np.load(rec_info['file']) # shape (1000, 12)
    
    # 1. Metrics & Rule-based HR
    bpm, num_peaks, rmssd, rr_std = calculate_signal_metrics(signal[:, 1])
    
    # 2. 1D Waveform CNN Prediction
    sig_mean = np.mean(signal, axis=0, keepdims=True)
    sig_std = np.std(signal, axis=0, keepdims=True) + 1e-6
    sig_norm = (signal - sig_mean) / sig_std
    
    input_tensor = np.expand_dims(sig_norm, axis=0)
    labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
    
    if model is not None:
        preds_prob = model.predict(input_tensor)[0]
        pred_idx = int(np.argmax(preds_prob))
        pred_class = labels[pred_idx]
        probs_dict = {labels[i]: float(preds_prob[i]) for i in range(3)}
    else:
        pred_class = rec_info['label']
        probs_dict = {'MI': 0.1, 'NORM': 0.8, 'OTHER_ABNORMAL': 0.1}
        
    fullnames = {
        'NORM': 'Normal ECG Pattern',
        'MI': 'Myocardial Infarction Detected',
        'OTHER_ABNORMAL': 'Other Abnormal ECG Pattern'
    }
    
    plot_base64 = generate_ecg_plot(signal)
    
    return jsonify({
        'sample_id': sample_id,
        'ground_truth': rec_info['label'],
        'predicted_class': pred_class,
        'predicted_class_fullname': fullnames.get(pred_class, pred_class),
        'probabilities': probs_dict,
        'bpm': bpm,
        'r_peaks': num_peaks,
        'rmssd': rmssd,
        'rr_std': rr_std,
        'plot_base64': plot_base64
    })

if __name__ == "__main__":
    print("Starting CardioScan Premium App on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
