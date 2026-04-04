from flask import Flask, request, jsonify
import os
from ecg_analysis import analyze_ecg

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CardioScan — ECG Disease Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Inter',sans-serif;background:#f7f7f5;color:#111;min-height:100vh}

/* NAV */
nav{background:#fff;border-bottom:1px solid #e8e8e8;position:sticky;top:0;z-index:100;height:56px;display:flex;align-items:center;padding:0 32px}
.nav-inner{max-width:1100px;margin:0 auto;width:100%;display:flex;align-items:center;gap:12px}
.nav-logo{width:32px;height:32px;border-radius:8px;background:#111;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.nav-logo svg{width:16px;height:16px;stroke:#fff;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.nav-name{font-size:16px;font-weight:700;letter-spacing:-0.3px}
.nav-tag{font-size:12px;color:#999;margin-left:4px;font-weight:400}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:24px}
.nav-link{font-size:13px;color:#666;text-decoration:none;font-weight:500}
.nav-link:hover{color:#111}
.nav-pill{font-size:12px;font-weight:600;background:#f0faf5;color:#0d6e42;border:1px solid #b7e5cf;padding:4px 12px;border-radius:99px}

/* HERO */
.hero{background:#fff;border-bottom:1px solid #e8e8e8;padding:64px 32px 56px}
.hero-inner{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:48px;align-items:center}
.hero-badge{display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 12px;border-radius:99px;background:#f0f0f0;border:1px solid #e0e0e0;font-size:11px;font-weight:600;color:#555;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:20px}
.hero-badge-dot{width:6px;height:6px;border-radius:50%;background:#1a9e60}
h1{font-size:40px;font-weight:700;letter-spacing:-1px;line-height:1.15;margin-bottom:16px;color:#111}
h1 span{color:#1a9e60}
.hero-desc{font-size:16px;color:#555;line-height:1.7;margin-bottom:28px;max-width:480px}
.hero-stats{display:flex;gap:28px}
.hstat-val{font-size:24px;font-weight:700;letter-spacing:-0.5px;color:#111}
.hstat-label{font-size:12px;color:#999;margin-top:2px}
.hero-right{display:flex;flex-direction:column;gap:12px}
.feat-card{background:#f7f7f5;border:1px solid #e8e8e8;border-radius:12px;padding:16px 18px;display:flex;align-items:flex-start;gap:12px}
.feat-icon{width:36px;height:36px;border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.feat-icon svg{width:18px;height:18px;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.fi-green{background:#f0faf5}
.fi-green svg{stroke:#1a9e60}
.fi-blue{background:#eff6ff}
.fi-blue svg{stroke:#2563eb}
.fi-amber{background:#fffbeb}
.fi-amber svg{stroke:#d97706}
.feat-title{font-size:13px;font-weight:600;margin-bottom:3px}
.feat-desc{font-size:12px;color:#888;line-height:1.5}

/* MAIN */
.main{max-width:1100px;margin:0 auto;padding:48px 32px 80px;display:grid;grid-template-columns:420px 1fr;gap:32px;align-items:start}

/* UPLOAD PANEL */
.panel{background:#fff;border:1px solid #e8e8e8;border-radius:16px;padding:28px;position:sticky;top:72px}
.panel-title{font-size:14px;font-weight:600;margin-bottom:20px;color:#111}
.dropzone{border:1.5px dashed #d4d4d4;border-radius:12px;padding:36px 20px;text-align:center;cursor:pointer;transition:all 0.15s;position:relative;background:#fafafa}
.dropzone:hover,.dropzone.over{border-color:#1a9e60;background:#f0faf5}
.dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dz-icon{width:44px;height:44px;border-radius:12px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;margin:0 auto 14px}
.dz-icon svg{width:20px;height:20px;stroke:#888;fill:none;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.dz-title{font-size:14px;font-weight:600;margin-bottom:4px;color:#111}
.dz-sub{font-size:12px;color:#aaa}
#fname{font-size:12px;color:#1a9e60;margin-top:10px;min-height:16px;font-weight:500}
#preview{display:none;width:100%;max-height:180px;object-fit:cover;border-radius:10px;margin-top:14px;border:1px solid #e8e8e8}
.btn-main{width:100%;margin-top:16px;height:44px;border-radius:10px;border:none;background:#111;color:#fff;font-size:14px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:all 0.15s;letter-spacing:-0.1px}
.btn-main:hover{background:#333}
.btn-main:disabled{opacity:0.3;cursor:not-allowed}
.spinner{width:15px;height:15px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.6s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
#err{display:none;margin-top:12px;padding:10px 14px;border-radius:8px;background:#fff1f1;border:1px solid #ffd4d4;font-size:12px;color:#c0392b}

/* how it works */
.how{margin-top:24px;padding-top:20px;border-top:1px solid #f0f0f0}
.how-title{font-size:11px;font-weight:600;color:#aaa;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:14px}
.step{display:flex;align-items:flex-start;gap:10px;margin-bottom:12px}
.step-num{width:22px;height:22px;border-radius:99px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#666;flex-shrink:0;margin-top:1px}
.step-text{font-size:12px;color:#666;line-height:1.5}
.step-text strong{color:#111;font-weight:600}

/* RESULTS PANEL */
#results{display:none}
.section-label{font-size:11px;font-weight:600;color:#aaa;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:12px}

/* diagnosis */
.diag-banner{border-radius:12px;padding:20px 22px;margin-bottom:20px;display:flex;gap:14px;align-items:flex-start}
.db-low{background:#f0faf5;border:1px solid #b7e5cf}
.db-moderate{background:#fffbf0;border:1px solid #f5dfa0}
.db-high{background:#fff5f5;border:1px solid #f5c0c0}
.db-unknown{background:#f7f7f7;border:1px solid #e0e0e0}
.diag-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:5px}
.db-low .diag-dot{background:#1a9e60}
.db-moderate .diag-dot{background:#c8960c}
.db-high .diag-dot{background:#d93025}
.db-unknown .diag-dot{background:#bbb}
.diag-head{display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap}
.diag-name{font-size:18px;font-weight:700;letter-spacing:-0.3px}
.risk-pill{height:22px;padding:0 10px;border-radius:99px;font-size:10px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;display:inline-flex;align-items:center}
.rp-low{background:#d4f5e5;color:#0d6e42}
.rp-moderate{background:#fef3cc;color:#8a6400}
.rp-high{background:#fde0e0;color:#991b1b}
.rp-unknown{background:#ebebeb;color:#777}
.diag-desc{font-size:13px;color:#555;line-height:1.65}
.source-tag{display:inline-flex;align-items:center;gap:5px;margin-top:10px;font-size:11px;color:#888;background:#f5f5f5;padding:3px 10px;border-radius:99px;border:1px solid #e8e8e8}
.source-dot{width:6px;height:6px;border-radius:50%;background:#1a9e60}

/* metrics */
.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:20px}
.metric{background:#f7f7f5;border-radius:10px;padding:16px}
.metric-label{font-size:10px;font-weight:600;color:#aaa;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px}
.metric-value{font-size:28px;font-weight:700;letter-spacing:-1px;line-height:1;color:#111}
.metric-unit{font-size:11px;color:#bbb;margin-top:4px;font-weight:500}
.metric-sub{font-size:11px;color:#aaa;margin-top:2px}

/* probs */
.probs-wrap{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:20px;margin-bottom:20px}
.probs-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.probs-title{font-size:13px;font-weight:600;color:#111}
.conf-badge{font-size:11px;font-weight:600;background:#f0faf5;color:#0d6e42;border:1px solid #b7e5cf;padding:3px 10px;border-radius:99px}
.prob-row{margin-bottom:14px}
.prob-top{display:flex;justify-content:space-between;margin-bottom:5px}
.prob-name{font-size:13px;font-weight:500;color:#111}
.prob-pct{font-size:13px;font-weight:600;color:#555}
.prob-track{height:6px;background:#f0f0f0;border-radius:99px;overflow:hidden}
.prob-fill-1{height:6px;border-radius:99px;background:#111;transition:width 0.6s ease}
.prob-fill-2{height:6px;border-radius:99px;background:#bbb;transition:width 0.6s ease}
.prob-fill-3{height:6px;border-radius:99px;background:#e0e0e0;transition:width 0.6s ease}
.rule-note{margin-top:14px;padding-top:12px;border-top:1px solid #f5f5f5;font-size:12px;color:#bbb;display:flex;align-items:center;gap:6px}

/* signal details */
.sig-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:20px}
.sig-cell{background:#f7f7f5;border-radius:8px;padding:10px 12px}
.sig-key{font-size:10px;font-weight:600;color:#aaa;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:4px}
.sig-val{font-size:13px;font-weight:600;color:#111}

/* disclaimer */
.disclaimer{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:16px 18px;display:flex;gap:12px;align-items:flex-start}
.disc-icon{width:32px;height:32px;border-radius:8px;background:#fffbeb;border:1px solid #f5dfa0;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.disc-icon svg{width:16px;height:16px;stroke:#d97706;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.disc-title{font-size:12px;font-weight:600;color:#111;margin-bottom:3px}
.disc-text{font-size:12px;color:#888;line-height:1.6}

/* FOOTER */
footer{background:#fff;border-top:1px solid #e8e8e8;padding:24px 32px;margin-top:auto}
.footer-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;justify-content:space-between}
.footer-left{font-size:12px;color:#aaa}
.footer-left strong{color:#555;font-weight:600}
.footer-tags{display:flex;gap:8px}
.ftag{font-size:11px;color:#999;background:#f5f5f5;padding:3px 10px;border-radius:99px;border:1px solid #ebebeb}

/* placeholder */
.results-placeholder{background:#fff;border:1px solid #e8e8e8;border-radius:16px;padding:60px 32px;text-align:center}
.ph-icon{width:56px;height:56px;border-radius:16px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;margin:0 auto 16px}
.ph-icon svg{width:24px;height:24px;stroke:#ccc;fill:none;stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.ph-title{font-size:15px;font-weight:600;color:#ccc;margin-bottom:6px}
.ph-sub{font-size:13px;color:#ddd}

@media(max-width:900px){
  .hero-inner{grid-template-columns:1fr}
  .hero-right{display:none}
  .main{grid-template-columns:1fr;padding:24px 16px 60px}
  .panel{position:static}
  .metrics{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nav-inner">
    <div class="nav-logo">
      <svg viewBox="0 0 24 24"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
    </div>
    <span class="nav-name">CardioScan</span>
    <span class="nav-tag">ECG Disease Detection</span>
    <div class="nav-right">
      <a class="nav-link" href="#upload">Upload</a>
      <a class="nav-link" href="#about">About</a>
      <span class="nav-pill">Final Year Project</span>
    </div>
  </div>
</nav>

<!-- HERO -->
<section class="hero">
  <div class="hero-inner">
    <div>
      <div class="hero-badge"><span class="hero-badge-dot"></span>AI-Powered Cardiac Analysis</div>
      <h1>Detect heart conditions from <span>ECG images</span></h1>
      <p class="hero-desc">CardioScan uses a trained machine learning model to analyse 12-lead ECG images and classify cardiac conditions including Normal Sinus Rhythm, Myocardial Infarction, and Arrhythmia.</p>
      <div class="hero-stats">
        <div>
          <div class="hstat-val">62%</div>
          <div class="hstat-label">Validation accuracy</div>
        </div>
        <div>
          <div class="hstat-val">1,592</div>
          <div class="hstat-label">Training images</div>
        </div>
        <div>
          <div class="hstat-val">3</div>
          <div class="hstat-label">Conditions detected</div>
        </div>
      </div>
    </div>
    <div class="hero-right">
      <div class="feat-card">
        <div class="feat-icon fi-green">
          <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        </div>
        <div>
          <div class="feat-title">Signal Analysis</div>
          <div class="feat-desc">Extracts R-peaks, computes BPM, RMSSD and HRV metrics from 12-lead ECG strips</div>
        </div>
      </div>
      <div class="feat-card">
        <div class="feat-icon fi-blue">
          <svg viewBox="0 0 24 24"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
        </div>
        <div>
          <div class="feat-title">SVM Classifier</div>
          <div class="feat-desc">Trained on PTB-XL dataset with gradient and projection features for image-based classification</div>
        </div>
      </div>
      <div class="feat-card">
        <div class="feat-icon fi-amber">
          <svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        </div>
        <div>
          <div class="feat-title">Multi-condition Detection</div>
          <div class="feat-desc">Classifies Normal, Myocardial Infarction (MI), and Abnormal/Arrhythmia conditions</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- MAIN -->
<section class="main" id="upload">

  <!-- LEFT: Upload -->
  <div>
    <div class="panel">
      <div class="panel-title">Upload ECG Image</div>
      <div class="dropzone" id="dz">
        <input type="file" id="fi" accept="image/*"/>
        <div class="dz-icon">
          <svg viewBox="0 0 24 24"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>
        </div>
        <div class="dz-title">Drop image here or click to browse</div>
        <div class="dz-sub">PNG · JPG · BMP · 12-lead or rhythm strip</div>
      </div>
      <div id="fname"></div>
      <img id="preview"/>
      <button class="btn-main" id="abtn" disabled onclick="run()">
        <span class="spinner" id="spin"></span>
        <span id="btxt">Analyse ECG</span>
      </button>
      <div id="err"></div>

      <div class="how">
        <div class="how-title">How it works</div>
        <div class="step"><div class="step-num">1</div><div class="step-text"><strong>Upload</strong> a 12-lead ECG image (standard paper or digital)</div></div>
        <div class="step"><div class="step-num">2</div><div class="step-text"><strong>Signal extraction</strong> detects lead rows and extracts waveform data</div></div>
        <div class="step"><div class="step-num">3</div><div class="step-text"><strong>SVM model</strong> classifies the condition from image features</div></div>
        <div class="step"><div class="step-num">4</div><div class="step-text"><strong>Results</strong> show diagnosis, confidence, and signal metrics</div></div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Results -->
  <div>
    <div id="results-placeholder" class="results-placeholder">
      <div class="ph-icon">
        <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      </div>
      <div class="ph-title">No analysis yet</div>
      <div class="ph-sub">Upload an ECG image to see results</div>
    </div>

    <div id="results">
      <div class="section-label" style="margin-bottom:14px">Analysis Report</div>

      <!-- Diagnosis -->
      <div class="diag-banner db-unknown" id="diag-banner">
        <div class="diag-dot"></div>
        <div style="flex:1">
          <div class="diag-head">
            <span class="diag-name" id="d-name">—</span>
            <span class="risk-pill rp-unknown" id="d-risk">—</span>
          </div>
          <div class="diag-desc" id="d-desc">—</div>
          <div class="source-tag" id="d-source"><span class="source-dot"></span><span id="d-source-txt">—</span></div>
        </div>
      </div>

      <!-- Metrics -->
      <div class="section-label">Signal metrics</div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Heart Rate</div>
          <div class="metric-value" id="m-bpm">—</div>
          <div class="metric-unit">BPM</div>
          <div class="metric-sub" id="m-bpm-note"></div>
        </div>
        <div class="metric">
          <div class="metric-label">R-Peaks</div>
          <div class="metric-value" id="m-peaks">—</div>
          <div class="metric-unit">Detected</div>
        </div>
        <div class="metric">
          <div class="metric-label">RMSSD</div>
          <div class="metric-value" id="m-rmssd">—</div>
          <div class="metric-unit">ms · HRV</div>
        </div>
        <div class="metric">
          <div class="metric-label">RR Std Dev</div>
          <div class="metric-value" id="m-rrstd">—</div>
          <div class="metric-unit">ms · Rhythm</div>
        </div>
      </div>

      <!-- Probabilities -->
      <div class="probs-wrap" id="probs-card" style="display:none">
        <div class="probs-head">
          <span class="probs-title">Model class probabilities</span>
          <span class="conf-badge" id="conf-badge">—</span>
        </div>
        <div id="prob-bars"></div>
        <div class="rule-note">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#bbb" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          Rule-based fallback: <strong id="rule-cond" style="color:#888;margin-left:4px">—</strong>
        </div>
      </div>

      <!-- Signal Details -->
      <div class="section-label">Signal details</div>
      <div class="sig-grid" id="sig-grid"></div>

      <!-- Disclaimer -->
      <div class="disclaimer" id="about">
        <div class="disc-icon">
          <svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <div>
          <div class="disc-title">Clinical Disclaimer</div>
          <div class="disc-text">This tool is developed for educational and research purposes as part of a final year project. Results must not replace a physician's interpretation or formal clinical ECG reading. Always consult a qualified cardiologist for medical diagnosis and treatment.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- FOOTER -->
<footer>
  <div class="footer-inner">
    <div class="footer-left">
      <strong>CardioScan</strong> · ECG Disease Detection · Final Year Project
    </div>
    <div class="footer-tags">
      <span class="ftag">PTB-XL Dataset</span>
      <span class="ftag">SVM Classifier</span>
      <span class="ftag">Flask + OpenCV</span>
    </div>
  </div>
</footer>

<script>
const fi=document.getElementById('fi'),dz=document.getElementById('dz'),abtn=document.getElementById('abtn');
let sel=null;

dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');const f=e.dataTransfer.files[0];if(f)setFile(f)});
fi.addEventListener('change',()=>{if(fi.files[0])setFile(fi.files[0])});

function setFile(f){
  sel=f;
  document.getElementById('fname').textContent=f.name;
  abtn.disabled=false;
  const r=new FileReader();
  r.onload=e=>{const p=document.getElementById('preview');p.src=e.target.result;p.style.display='block'};
  r.readAsDataURL(f);
  document.getElementById('results').style.display='none';
  document.getElementById('results-placeholder').style.display='block';
  document.getElementById('err').style.display='none';
}

async function run(){
  if(!sel)return;
  abtn.disabled=true;
  document.getElementById('spin').style.display='block';
  document.getElementById('btxt').textContent='Analysing…';
  document.getElementById('err').style.display='none';
  const fd=new FormData();fd.append('file',sel);
  try{
    const res=await fetch('/predict',{method:'POST',body:fd});
    if(!res.ok)throw new Error('Server error '+res.status);
    render(await res.json());
  }catch(e){
    const el=document.getElementById('err');
    el.textContent='Analysis failed: '+(e.message||'Unknown error');
    el.style.display='block';
  }finally{
    abtn.disabled=false;
    document.getElementById('spin').style.display='none';
    document.getElementById('btxt').textContent='Analyse ECG';
  }
}

function render(d){
  document.getElementById('results-placeholder').style.display='none';
  document.getElementById('results').style.display='block';

  // scroll to results on mobile
  if(window.innerWidth<900) document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});

  const risk=(d.risk||'unknown').toLowerCase();
  const banner=document.getElementById('diag-banner');
  banner.className='diag-banner db-'+(risk==='low'?'low':risk==='moderate'?'moderate':risk==='high'?'high':'unknown');
  document.getElementById('d-name').textContent=d.condition||'—';
  document.getElementById('d-desc').textContent=d.description||'—';
  const rp=document.getElementById('d-risk');
  rp.textContent=(d.risk||'Unknown')+' Risk';
  rp.className='risk-pill rp-'+(risk==='low'?'low':risk==='moderate'?'moderate':risk==='high'?'high':'unknown');
  document.getElementById('d-source-txt').textContent=d.source==='svm_model'?'SVM Model — PTB-XL trained':'Rule-based signal analysis';

  // metrics
  const bpm=d.bpm||0;
  document.getElementById('m-bpm').textContent=bpm||'—';
  document.getElementById('m-bpm-note').textContent=bpm>0?(bpm<60?'Below normal':bpm>100?'Above normal':'Normal range'):'';
  document.getElementById('m-peaks').textContent=d.peaks||'—';
  document.getElementById('m-rmssd').textContent=d.rmssd!==undefined?Math.round(d.rmssd):'—';
  document.getElementById('m-rrstd').textContent=d.rr_std_ms!==undefined?Math.round(d.rr_std_ms):'—';

  // probs
  const pc=document.getElementById('probs-card');
  if(d.all_probs&&Object.keys(d.all_probs).length>0){
    pc.style.display='block';
    document.getElementById('conf-badge').textContent=(d.cnn_confidence||'—')+'% confidence';
    document.getElementById('rule-cond').textContent=(d.rule_condition||'—')+' ('+( d.rule_risk||'—')+')';
    const sorted=Object.entries(d.all_probs).sort((a,b)=>b[1]-a[1]);
    document.getElementById('prob-bars').innerHTML=sorted.map(([name,pct],i)=>`
      <div class="prob-row">
        <div class="prob-top"><span class="prob-name">${name}</span><span class="prob-pct">${pct.toFixed(1)}%</span></div>
        <div class="prob-track"><div class="prob-fill-${i+1}" style="width:${pct}%"></div></div>
      </div>`).join('');
  }else{pc.style.display='none'}

  // signal details
  const meta=[
    ['Signal quality', d.signal_quality||'—'],
    ['Confidence', d.confidence||'—'],
    ['Est. sampling rate', (d.fs_estimated||'—')+' px/s'],
    ['Layout detected', d.layout_detected||'—'],
    ['Lead used', d.lead_used||'—'],
    ['Leads found', (d.leads_found||[]).length>0?(d.leads_found||[]).join(', '):'—'],
  ];
  document.getElementById('sig-grid').innerHTML=meta.map(([k,v])=>`
    <div class="sig-cell"><div class="sig-key">${k}</div><div class="sig-val">${v}</div></div>`).join('');

  document.getElementById('results').style.display='block';
}
</script>
</body>
</html>"""

@app.route("/")
def home():
    return HTML

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    filepath = os.path.join("static", file.filename)
    file.save(filepath)
    try:
        result = analyze_ecg(filepath)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("CardioScan starting...")
    app.run(debug=True)