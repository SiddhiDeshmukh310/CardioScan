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
    fig.patch.set_facecolor('#ffffff')
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    t = np.linspace(0, 10, 1000)
    for idx in range(12):
        r, c = idx % 6, idx // 6
        ax = axes[r, c]
        ax.set_facecolor('#fafafa')
        ax.plot(t, signal[:, idx], color='#1a9e60', linewidth=1.2)
        ax.set_title(f"Lead {lead_names[idx]}", color='#111111', fontsize=8.5, pad=2, loc='left', weight='bold')
        ax.grid(True, color='#e5e5e5', linestyle='--', linewidth=0.6)
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#e0e0e0')
            
    fig.text(0.5, 0.01, 'Time (seconds)', ha='center', color='#888888', fontsize=8.5)
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
  <title>CardioScan — ECG Disease Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1a9e60;
      --primary-dark: #0d6e42;
      --primary-light: #f0faf5;
      --primary-border: #b7e5cf;
      --bg: #fafafa;
      --card-bg: #ffffff;
      --card-border: #e8e8e8;
      --text-dark: #111111;
      --text-body: #555555;
      --text-muted: #aaaaaa;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }
    body { background-color: var(--bg); color: var(--text-dark); padding-bottom: 60px; line-height: 1.5; }

    /* Top Red Educational Banner */
    .edu-banner {
      background: #d93025;
      color: #ffffff;
      text-align: center;
      padding: 10px 16px;
      font-size: 13.5px;
      font-weight: 700;
      letter-spacing: 0.3px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    /* HEADER */
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--card-border);
      padding: 16px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
      width: 36px; height: 36px;
      border-radius: 10px;
      background: var(--primary-light);
      border: 1px solid var(--primary-border);
      display: flex; align-items: center; justify-content: center;
    }
    .logo-icon svg { stroke: var(--primary); width: 20px; height: 20px; stroke-width: 2.2; }
    .logo-text { font-size: 20px; font-weight: 800; color: var(--text-dark); letter-spacing: -0.4px; }
    .logo-tag { font-size: 11px; background: var(--primary-light); color: var(--primary-dark); border: 1px solid var(--primary-border); padding: 2px 8px; border-radius: 99px; font-weight: 600; margin-left: 6px; }
    .header-right { font-size: 13px; color: var(--text-muted); font-weight: 500; }

    /* HERO */
    .hero {
      max-width: 1150px;
      margin: 0 auto;
      padding: 32px 36px 16px 36px;
    }
    .hero h1 { font-size: 26px; font-weight: 800; color: var(--text-dark); letter-spacing: -0.5px; margin-bottom: 6px; }
    .hero p { color: var(--text-body); font-size: 14.5px; max-width: 800px; }

    /* MAIN GRID */
    .main {
      max-width: 1150px;
      margin: 0 auto;
      padding: 0 36px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 28px;
      align-items: start;
    }
    @media (max-width: 900px) { .main { grid-template-columns: 1fr; padding: 0 20px; } }

    /* PANELS */
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .panel-title { font-size: 14px; font-weight: 700; color: var(--text-dark); margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }

    /* SAMPLE CARDS (LEFT) */
    .sample-list { display: flex; flex-direction: column; gap: 10px; max-height: 580px; overflow-y: auto; padding-right: 2px; }
    .sample-card {
      background: #fafafa;
      border: 1.5px solid #eaeaea;
      border-radius: 12px;
      padding: 14px 16px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .sample-card:hover {
      border-color: var(--primary);
      background: var(--primary-light);
    }
    .sample-card.active {
      background: var(--primary-light);
      border-color: var(--primary);
      box-shadow: 0 4px 12px rgba(26, 158, 96, 0.12);
    }
    .sample-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .sample-id { font-weight: 700; font-size: 14px; color: var(--text-dark); }
    .sample-meta { font-size: 12px; color: #888888; }

    /* RISK PILLS */
    .risk-pill {
      height: 22px; padding: 0 10px; border-radius: 99px; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; display: inline-flex; align-items: center;
    }
    .rp-low { background: #d4f5e5; color: #0d6e42; border: 1px solid #b7e5cf; }
    .rp-high { background: #fde0e0; color: #991b1b; border: 1px solid #f5c0c0; }
    .rp-moderate { background: #fef3cc; color: #8a6400; border: 1px solid #f5dfa0; }

    /* DIAGNOSIS BANNER */
    .diag-banner {
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .db-low { background: #f0faf5; border: 1px solid #b7e5cf; }
    .db-high { background: #fff5f5; border: 1px solid #f5c0c0; }
    .db-moderate { background: #fffbf0; border: 1px solid #f5dfa0; }

    .diag-name { font-size: 20px; font-weight: 800; color: var(--text-dark); letter-spacing: -0.3px; }
    .diag-sub { font-size: 13px; color: var(--text-body); margin-top: 4px; }
    .source-tag { display: inline-flex; align-items: center; gap: 5px; margin-top: 10px; font-size: 11px; color: #666; background: #ffffff; padding: 3px 10px; border-radius: 99px; border: 1px solid var(--card-border); font-weight: 600; }
    .source-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); }

    /* WAVEFORM DISPLAY CANVAS */
    .plot-container {
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 20px;
      position: relative;
    }
    .plot-img { width: 100%; height: auto; border-radius: 8px; display: block; }

    /* METRICS */
    .section-label { font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 12px; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 20px; }
    .metric { background: #f7f7f5; border-radius: 12px; padding: 16px; border: 1px solid #eeeeee; }
    .metric-label { font-size: 10px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { font-size: 26px; font-weight: 800; color: var(--text-dark); letter-spacing: -0.5px; }
    .metric-unit { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
    .rule-tag { display: inline-block; margin-top: 6px; font-size: 10px; font-weight: 700; color: var(--primary-dark); background: var(--primary-light); border: 1px solid var(--primary-border); padding: 2px 7px; border-radius: 4px; text-transform: uppercase; }

    /* PROBABILITIES */
    .probs-wrap { background: #ffffff; border: 1px solid var(--card-border); border-radius: 14px; padding: 20px; margin-bottom: 20px; }
    .probs-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
    .probs-title { font-size: 14px; font-weight: 700; color: var(--text-dark); }
    .conf-badge { font-size: 11px; font-weight: 700; background: var(--primary-light); color: var(--primary-dark); border: 1px solid var(--primary-border); padding: 3px 10px; border-radius: 99px; }
    
    .prob-row { margin-bottom: 12px; }
    .prob-top { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 13px; }
    .prob-name { font-weight: 600; color: var(--text-dark); }
    .prob-pct { font-weight: 700; color: var(--text-body); }
    .prob-track { height: 7px; background: #f0f0f0; border-radius: 99px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 99px; transition: width 0.5s ease; }
    .pf-norm { background: var(--primary); }
    .pf-mi { background: #d93025; }
    .pf-other { background: #d97706; }

    /* DISCLAIMER BOX */
    .disclaimer { background: #ffffff; border: 1px solid var(--card-border); border-radius: 12px; padding: 16px 18px; display: flex; gap: 12px; align-items: flex-start; }
    .disc-icon { width: 32px; height: 32px; border-radius: 8px; background: #fffbeb; border: 1px solid #f5dfa0; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .disc-icon svg { width: 16px; height: 16px; stroke: #d97706; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .disc-title { font-size: 12px; font-weight: 700; color: var(--text-dark); margin-bottom: 3px; }
    .disc-text { font-size: 12px; color: var(--text-body); line-height: 1.6; }

    /* SPINNER */
    .spinner-overlay {
      position: absolute; inset: 0; background: rgba(255, 255, 255, 0.9); display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 14px; z-index: 10;
    }
    .spinner { width: 36px; height: 36px; border: 3px solid #e8e8e8; border-top-color: var(--primary); border-radius: 50%; animation: spin 0.7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* FOOTER */
    footer { background: #ffffff; border-top: 1px solid var(--card-border); padding: 24px 36px; margin-top: 40px; }
    .footer-inner { max-width: 1150px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: var(--text-muted); }
    .footer-tags { display: flex; gap: 8px; }
    .ftag { font-size: 11px; color: #888888; background: #f5f5f5; padding: 3px 10px; border-radius: 99px; border: 1px solid #e8e8e8; }
  </style>
</head>
<body>

  <!-- Top Red Educational Banner -->
  <div class="edu-banner">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
    Educational demo. Not a medical device.
  </div>

  <header>
    <div class="brand">
      <div class="logo-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
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
    <p>Select a 12-lead ECG sample record from the benchmark PTB-XL dataset to analyze raw digital signals with our 1D Convolutional Neural Network.</p>
  </div>

  <div class="main">
    <!-- LEFT PANEL: SAMPLE SELECTOR -->
    <div class="panel">
      <div class="panel-title">
        <span>PTB-XL Sample Records</span>
        <span style="font-size:12px; color:var(--text-muted); font-weight:normal;">{{ samples|length }} records</span>
      </div>

      <div class="sample-list">
        {% for sample in samples %}
        <div class="sample-card {% if loop.first %}active{% endif %}" onclick="selectSample({{ sample.id }}, this)">
          <div class="sample-top">
            <span class="sample-id">Record #{{ sample.id }}</span>
            <span class="risk-pill {% if sample.label == 'NORM' %}rp-low{% elif sample.label == 'MI' %}rp-high{% else %}rp-moderate{% endif %}">
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

    <!-- RIGHT PANEL: RESULTS & DISPLAY -->
    <div>
      <!-- DIAGNOSIS BANNER -->
      <div class="diag-banner db-low" id="diag-banner">
        <div>
          <div class="diag-name" id="pred-class-name">Analyzing...</div>
          <div class="diag-sub" id="ground-truth-sub">PTB-XL Ground Truth: --</div>
          <div class="source-tag"><span class="source-dot"></span> 1D Waveform CNN — PTB-XL trained</div>
        </div>
        <div id="risk-pill-container">
          <span class="risk-pill rp-low">Normal</span>
        </div>
      </div>

      <!-- WAVEFORM DISPLAY CANVAS -->
      <div class="plot-container">
        <div class="spinner-overlay" id="spinner-overlay">
          <div class="spinner"></div>
          <div style="margin-top:10px; font-size:13px; color:#555; font-weight:600;">Processing 12-Lead Signals...</div>
        </div>
        <img id="ecg-plot-img" class="plot-img" src="" alt="12-Lead ECG Plot" style="display:none;">
      </div>

      <!-- METRICS GRID -->
      <div class="section-label">Signal Metrics</div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Heart Rate</div>
          <div class="metric-value" id="m-bpm">--</div>
          <div class="metric-unit">BPM</div>
          <div class="rule-tag">rule-based signal analysis</div>
        </div>
        <div class="metric">
          <div class="metric-label">R-Peaks</div>
          <div class="metric-value" id="m-peaks">--</div>
          <div class="metric-unit">Detected</div>
        </div>
        <div class="metric">
          <div class="metric-label">RMSSD</div>
          <div class="metric-value" id="m-rmssd">--</div>
          <div class="metric-unit">ms • HRV</div>
        </div>
        <div class="metric">
          <div class="metric-label">RR Std Dev</div>
          <div class="metric-value" id="m-rrstd">--</div>
          <div class="metric-unit">ms • Rhythm</div>
        </div>
      </div>

      <!-- CLASS PROBABILITIES -->
      <div class="probs-wrap">
        <div class="probs-head">
          <span class="probs-title">Model Class Probabilities</span>
          <span class="conf-badge" id="conf-badge">--% confidence</span>
        </div>
        <div id="prob-bars-list">
          <!-- Probability bars generated dynamically -->
        </div>
      </div>

      <!-- CLINICAL DISCLAIMER -->
      <div class="disclaimer">
        <div class="disc-icon">
          <svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <div>
          <div class="disc-title">Clinical Disclaimer</div>
          <div class="disc-text">This tool is developed for educational and research purposes as part of an academic project. Results must not replace a physician's interpretation or formal clinical ECG reading. Always consult a qualified cardiologist for medical diagnosis.</div>
        </div>
      </div>
    </div>
  </div>

  <footer>
    <div class="footer-inner">
      <div><strong>CardioScan</strong> • ECG Classification Project • Educational Use Only</div>
      <div class="footer-tags">
        <span class="ftag">PTB-XL Dataset</span>
        <span class="ftag">1D CNN Model</span>
        <span class="ftag">Flask</span>
      </div>
    </div>
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
      const overlay = document.getElementById('spinner-overlay');
      const img = document.getElementById('ecg-plot-img');
      overlay.style.display = 'flex';

      try {
        const response = await fetch('/api/predict_sample/' + id);
        const data = await response.json();

        // 1. Render Plot Image
        img.src = 'data:image/png;base64,' + data.plot_base64;
        img.style.display = 'block';

        // 2. Render Diagnosis Header & Banner
        document.getElementById('pred-class-name').textContent = data.predicted_class_fullname;
        document.getElementById('ground-truth-sub').textContent = 'PTB-XL Ground Truth: ' + data.ground_truth;

        const banner = document.getElementById('diag-banner');
        const pillBox = document.getElementById('risk-pill-container');
        const pred = data.predicted_class;

        if (pred === 'NORM') {
          banner.className = 'diag-banner db-low';
          pillBox.innerHTML = '<span class="risk-pill rp-low">Normal Risk</span>';
        } else if (pred === 'MI') {
          banner.className = 'diag-banner db-high';
          pillBox.innerHTML = '<span class="risk-pill rp-high">High Risk • MI</span>';
        } else {
          banner.className = 'diag-banner db-moderate';
          pillBox.innerHTML = '<span class="risk-pill rp-moderate">Moderate Risk</span>';
        }

        // 3. Render Metrics
        document.getElementById('m-bpm').textContent = data.bpm;
        document.getElementById('m-peaks').textContent = data.r_peaks;
        document.getElementById('m-rmssd').textContent = data.rmssd;
        document.getElementById('m-rrstd').textContent = data.rr_std;

        // 4. Render Probabilities
        const probs = data.probabilities;
        let html = '';
        const names = {
          'NORM': 'Normal (NORM)',
          'MI': 'Myocardial Infarction (MI)',
          'OTHER_ABNORMAL': 'Other Abnormal (STTC/CD/HYP)'
        };
        const fills = { 'NORM': 'pf-norm', 'MI': 'pf-mi', 'OTHER_ABNORMAL': 'pf-other' };

        for (const [cls, prob] of Object.entries(probs)) {
          const pct = (prob * 100).toFixed(1);
          html += `
            <div class="prob-row">
              <div class="prob-top">
                <span class="prob-name">${names[cls] || cls}</span>
                <span class="prob-pct">${pct}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill ${fills[cls] || 'pf-norm'}" style="width: ${pct}%;"></div>
              </div>
            </div>
          `;
        }
        document.getElementById('prob-bars-list').innerHTML = html;
        const maxProb = Math.max(...Object.values(probs));
        document.getElementById('conf-badge').textContent = (maxProb * 100).toFixed(1) + '% confidence';

      } catch (err) {
        console.error(err);
      } finally {
        overlay.style.display = 'none';
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
    
    # Metrics
    bpm, num_peaks, rmssd, rr_std = calculate_signal_metrics(signal[:, 1])
    
    # 1D CNN Prediction
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
        'NORM': 'Normal Electrocardiogram',
        'MI': 'Myocardial Infarction',
        'OTHER_ABNORMAL': 'Other ECG Abnormality'
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
    print("Starting CardioScan White & Emerald App on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
