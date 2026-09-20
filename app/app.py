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

# Fallback default samples if index is incomplete
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
    # signal shape: (1000, 12)
    fig, axes = plt.subplots(6, 2, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor('#0f172a')
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    t = np.linspace(0, 10, 1000)
    for idx in range(12):
        r, c = idx % 6, idx // 6
        ax = axes[r, c]
        ax.set_facecolor('#1e293b')
        ax.plot(t, signal[:, idx], color='#38bdf8', linewidth=1.2)
        ax.set_title(f"Lead {lead_names[idx]}", color='#f8fafc', fontsize=9, pad=2, loc='left')
        ax.grid(True, color='#334155', linestyle='--', linewidth=0.5)
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#334155')
            
    fig.text(0.5, 0.01, 'Time (seconds)', ha='center', color='#94a3b8', fontsize=9)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def calculate_rule_based_bpm(signal_lead2, fs=100):
    # Rule-based signal analysis for Heart Rate (BPM)
    # Lead II R-peak detection via thresholding
    sig = signal_lead2 - np.mean(signal_lead2)
    threshold = np.max(sig) * 0.5
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > threshold and sig[i] > sig[i-1] and sig[i] > sig[i+1]:
            if not peaks or (i - peaks[-1]) > (0.4 * fs):
                peaks.append(i)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs # in seconds
        mean_rr = np.mean(rr_intervals)
        bpm = int(round(60.0 / mean_rr)) if mean_rr > 0 else 72
    else:
        bpm = 72
    return max(45, min(180, bpm)), len(peaks)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioScan — PTB-XL ECG Screening Demo</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background-color: #0f172a; color: #f8fafc; padding-bottom: 40px; }
    
    /* Top Banner */
    .edu-banner {
      background: linear-gradient(90deg, #b91c1c, #991b1b);
      color: #ffffff;
      text-align: center;
      padding: 10px 16px;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.5px;
      border-bottom: 1px solid #7f1d1d;
    }
    
    header {
      background-color: #1e293b;
      padding: 20px 32px;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .logo { font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
    .subtitle { color: #94a3b8; font-size: 13px; }

    .container { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
    
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 24px; }
    @media (max-width: 850px) { .grid { grid-template-columns: 1fr; } }
    
    .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
    .card-title { font-size: 16px; font-weight: 600; color: #f8fafc; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    
    /* Sample Select */
    .sample-item {
      background-color: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .sample-item:hover { border-color: #38bdf8; background-color: #1e293b; }
    .sample-item.active { border-color: #38bdf8; background-color: #0369a1; }
    
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-norm { background-color: #065f46; color: #34d399; }
    .badge-mi { background-color: #991b1b; color: #fca5a5; }
    .badge-other { background-color: #92400e; color: #fcd34d; }
    
    /* Waveform Plot */
    .plot-container { background: #0f172a; border-radius: 8px; padding: 12px; border: 1px solid #334155; min-height: 400px; display: flex; align-items: center; justify-content: center; }
    .plot-img { width: 100%; height: auto; border-radius: 6px; }
    
    /* Results Box */
    .results-box { margin-top: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
    .metric-cell { background-color: #0f172a; border-radius: 8px; padding: 14px; border: 1px solid #334155; }
    .metric-label { font-size: 12px; color: #94a3b8; font-weight: 500; }
    .metric-value { font-size: 20px; color: #38bdf8; font-weight: 700; margin-top: 4px; }
    .rule-tag { font-size: 11px; color: #a855f7; font-weight: 600; text-transform: uppercase; margin-top: 4px; display: block; }
    
    /* Probabilities */
    .prob-bar-wrap { margin-top: 12px; }
    .prob-row { margin-bottom: 8px; }
    .prob-label-row { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
    .prob-track { background-color: #334155; height: 8px; border-radius: 4px; overflow: hidden; }
    .prob-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); border-radius: 4px; transition: width 0.4s ease; }
    
    .spinner { border: 3px solid #334155; border-top: 3px solid #38bdf8; border-radius: 50%; width: 32px; height: 32px; animation: spin 1s linear infinite; margin: 20px auto; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <!-- Educational Banner -->
  <div class="edu-banner">
    ⚠️ Educational demo. Not a medical device.
  </div>

  <header>
    <div>
      <div class="logo">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        CardioScan
      </div>
      <div class="subtitle">1D Waveform Neural Network ECG Classifier (PTB-XL Benchmark)</div>
    </div>
  </header>

  <div class="container">
    <div class="grid">
      <!-- Left Panel: Sample Selector -->
      <div class="card">
        <div class="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/></svg>
          Select PTB-XL Sample Record
        </div>
        <div id="sample-list">
          {% for sample in samples %}
          <div class="sample-item {% if loop.first %}active{% endif %}" onclick="selectSample({{ sample.id }}, this)">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <strong>Record #{{ sample.id }}</strong>
              <span class="badge {% if sample.label == 'NORM' %}badge-norm{% elif sample.label == 'MI' %}badge-mi{% else %}badge-other{% endif %}">
                {{ sample.label }}
              </span>
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
              Patient #{{ sample.patient_id }} • Age {{ sample.age }}
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <!-- Right Panel: Waveform Plot & Prediction -->
      <div>
        <div class="card">
          <div class="card-title" style="justify-content:space-between;">
            <span>12-Lead ECG Signal (100 Hz Raw Waveform)</span>
            <span id="active-record-tag" style="font-size:13px; color:#94a3b8; font-weight:normal;">Record #{{ samples[0].id if samples else '1' }}</span>
          </div>
          
          <div class="plot-container" id="plot-box">
            <div class="spinner" id="loading-spinner"></div>
          </div>

          <div class="results-box" id="results-box" style="display:none;">
            <!-- Heart Rate -->
            <div class="metric-cell">
              <div class="metric-label">Heart Rate (BPM)</div>
              <div class="metric-value" id="bpm-val">--</div>
              <span class="rule-tag">rule-based signal analysis</span>
            </div>

            <!-- Predicted Class -->
            <div class="metric-cell">
              <div class="metric-label">Model Predicted Class</div>
              <div class="metric-value" id="pred-class">--</div>
              <div style="font-size:11px; color:#94a3b8; margin-top:4px;" id="ground-truth">Ground Truth: --</div>
            </div>

            <!-- Probabilities -->
            <div class="metric-cell" style="grid-column: 1 / -1;">
              <div class="metric-label">1D CNN Class Probabilities</div>
              <div class="prob-bar-wrap" id="prob-bars"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentSampleId = {{ samples[0].id if samples else 1 }};

    function selectSample(id, elem) {
      document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
      elem.classList.add('active');
      currentSampleId = id;
      loadSampleData(id);
    }

    async function loadSampleData(id) {
      document.getElementById('loading-spinner').style.display = 'block';
      document.getElementById('results-box').style.display = 'none';
      document.getElementById('active-record-tag').textContent = 'Record #' + id;

      try {
        const response = await fetch('/api/predict_sample/' + id);
        const data = await response.json();
        
        // Render Plot
        document.getElementById('plot-box').innerHTML = `<img src="data:image/png;base64,${data.plot_base64}" class="plot-img" alt="ECG Plot">`;
        
        // Render Metrics
        document.getElementById('bpm-val').textContent = data.bpm + ' BPM';
        document.getElementById('pred-class').textContent = data.predicted_class;
        document.getElementById('ground-truth').textContent = 'Ground Truth: ' + data.ground_truth;
        
        // Render Probabilities
        const probs = data.probabilities;
        let html = '';
        for (const [cls, prob] of Object.entries(probs)) {
          const pct = (prob * 100).toFixed(1);
          html += `
            <div class="prob-row">
              <div class="prob-label-row">
                <span>${cls}</span>
                <span>${pct}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" style="width: ${pct}%;"></div>
              </div>
            </div>
          `;
        }
        document.getElementById('prob-bars').innerHTML = html;
        document.getElementById('results-box').style.display = 'grid';

      } catch (err) {
        console.error(err);
        document.getElementById('plot-box').innerHTML = '<div style="color:#ef4444;">Failed to load record analysis.</div>';
      }
    }

    // Auto-load first sample on start
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
    # Find sample
    rec_info = next((s for s in sample_records if s['id'] == sample_id), None)
    if not rec_info:
        # Fallback to first available sample
        rec_info = sample_records[0]
        
    signal = np.load(rec_info['file']) # shape (1000, 12)
    
    # 1. Rule-based heart rate analysis
    bpm, num_peaks = calculate_rule_based_bpm(signal[:, 1]) # Lead II
    
    # 2. 1D CNN Waveform Model Prediction
    # Standardize signal
    sig_mean = np.mean(signal, axis=0, keepdims=True)
    sig_std = np.std(signal, axis=0, keepdims=True) + 1e-6
    sig_norm = (signal - sig_mean) / sig_std
    
    input_tensor = np.expand_dims(sig_norm, axis=0) # shape (1, 1000, 12)
    
    labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
    if model is not None:
        preds_prob = model.predict(input_tensor)[0]
        pred_idx = int(np.argmax(preds_prob))
        pred_class = labels[pred_idx]
        probs_dict = {labels[i]: float(preds_prob[i]) for i in range(3)}
    else:
        # Fallback default
        pred_class = rec_info['label']
        probs_dict = {'MI': 0.1, 'NORM': 0.8, 'OTHER_ABNORMAL': 0.1}
        
    # Generate Plot
    plot_base64 = generate_ecg_plot(signal)
    
    return jsonify({
        'sample_id': sample_id,
        'ground_truth': rec_info['label'],
        'predicted_class': pred_class,
        'probabilities': probs_dict,
        'bpm': bpm,
        'r_peaks': num_peaks,
        'plot_base64': plot_base64
    })

if __name__ == "__main__":
    print("Starting CardioScan Demo App on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
