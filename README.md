# CardioScan: ECG Classification on PTB-XL (Image vs Waveform Models)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/tensorflow-2.21-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CardioScan is an educational research project evaluating deep learning approaches for 12-lead Electrocardiogram (ECG) classification on the publicly available **PTB-XL dataset**.

> [!IMPORTANT]
> **Clinical Disclaimer**: Educational and research purposes only. Not a certified medical device.

---

## Overview & Key Findings

This project systematically diagnoses and compares two primary modeling approaches for ECG classification under a strict **patient-wise leak-free split**:

1. **2D Image Model (EfficientNetB0)**: Evaluated on the image subset test split (N=239). Under a leak-free patient split, the 2D image model collapsed to predicting the majority class (NORM) for all test images (**Accuracy: 0.5565**, **Macro F1: 0.2384**, **Macro AUROC: 0.5061**), matching majority class rates and failing to beat baselines. Note: Legacy leaky model artifacts (est_efficientnet.h5 and est_efficientnet_leakfree.h5) have been removed.
2. **1D Waveform Model (1D CNN)**: Retrained on ALL usable PTB-XL records across folds 1-8 (16,303 training records), validated on fold 9 (2,034 validation records), and evaluated on the full test fold 10 (N=2,050). The 1D CNN achieved strong, statistically significant performance (**Accuracy: 0.7644**, **Macro F1: 0.7383**, **Macro AUROC: 0.9005**), with a bootstrap 95% confidence interval strictly above all baselines.

---

## Final Performance Comparison Table

Evaluated on the leak-free test split (**PTB-XL Stratified Fold 10**):

| Model / Strategy | Test Samples (N) | Test Accuracy (95% CI) | Macro F1 (95% CI) | Macro AUROC (95% CI) | MI Recall | NORM Recall | OTHER Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline (Always Predict NORM)** | 2,050 | 0.4434 [0.4215, 0.4629] | 0.2048 [0.1977, 0.2110] | 0.5000 [0.5000, 0.5000] | 0.0000 (0/415) | 1.0000 (909/909) | 0.0000 (0/726) | Baseline |
| **Random Class Frequency Baseline** | 2,050 | 0.3624 [0.3405, 0.3844] | 0.3321 [0.3122, 0.3534] | 0.5000 [0.5000, 0.5000] | 0.2458 (102/415) | 0.4851 (441/909) | 0.2879 (209/726) | Baseline |
| **2D Image Model (EfficientNetB0)** | 239* | 0.5565 [0.4937, 0.6192] | 0.2384 [0.2201, 0.2547] | 0.5061 [0.4610, 0.5512] | 0.0000 (0/42) | 1.0000 (133/133) | 0.0000 (0/64) | Majority Collapse (Evaluated on Image Split) |
| **1D Waveform Model (1D CNN)** | 2,050 | **0.7644** [0.7449, 0.7824] | **0.7383** [0.7175, 0.7582] | **0.9005** [0.8891, 0.9104] | **0.6289** (261/415) | **0.9153** (832/909) | **0.6529** (474/726) | **PASSED (Beats baseline; 95% CIs strictly non-overlapping)** |

*Note: The 2D image model was evaluated on the N=239 image test split. Per-class supports add up to N in all rows (415 + 909 + 726 = 2,050 for full test fold; 42 + 133 + 64 = 239 for image split).

---

## Architecture & Pipeline Diagram

`mermaid
graph TD
    A[PTB-XL Raw Records 100Hz & Metadata] --> B[Ground Truth Relabeling: MI > NORM > OTHER_ABNORMAL]
    B --> C[Patient-wise Stratified Split: strat_fold 1-10]
    C -->|Folds 1-8: 16,303 records| D[Train Fold]
    C -->|Fold 9: 2,034 records| E[Validation Fold]
    C -->|Fold 10: 2,050 records| F[Test Fold]
    D --> G[2D Image EfficientNet Pipeline]
    D --> H[1D Waveform CNN Pipeline]
    F -->|Single Evaluation N=239| I[Image Model: Macro F1 0.2384 - Predicts NORM Only]
    F -->|Single Evaluation N=2050| J[Waveform Model: Macro F1 0.7383, AUROC 0.9005 - Passed]
    J --> K[Flask Interactive Web Demo app/app.py]
`

---

## Tech Stack

- **Core & Logic**: Python 3.11, NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow 2.21, Keras
- **ECG Signal Processing**: WFDB (wfdb), Matplotlib
- **Web Application**: Flask, HTML5, CSS3, JavaScript (Vanilla)

---

## Repository Structure

`
CardioScan/
├── app/
│   ├── app.py                  # Flask web application demo (sample picker & waveform 1D model)
│   └── samples/                # Included PTB-XL sample records (.npy signals & metadata)
├── docs/                       # Screenshots and user flow documentation
├── model/
│   └── waveform_1d_cnn.h5      # Trained 1D waveform CNN model (All usable PTB-XL records)
├── results/
│   ├── predictions_waveform.csv# Saved predictions for full test fold 10 (N=2,050)
│   ├── predictions_image.csv   # Saved predictions for image test fold (N=239)
│   ├── metrics_waveform.json   # Evaluation metrics and 95% bootstrap CIs
│   └── confusion_matrix_waveform.png
├── src/
│   ├── train_waveform.py       # Trains 1D waveform CNN on folds 1-8, tunes on fold 9
│   ├── evaluate_waveform.py    # Evaluates 1D CNN on full fold 10 test set (N=2,050)
│   └── download_ptbxl_100hz.py # Multi-threaded PhysioNet PTB-XL downloader
├── tests/
│   └── test_leak_free_split.py # Pytest unit tests for patient & image split integrity
├── splits/
│   └── split.csv               # Tracked leak-free split index
├── ptbxl_database.csv          # PTB-XL database metadata
├── scp_statements.csv          # SCP statement definitions
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
`

---

## Data & Leak-Free Split Methodology

### Ground Truth Relabeling
Labels are derived from official SCP statements (scp_statements.csv) with likelihood >= 50.0:
1. **MI**: Assigned if Myocardial Infarction (MI) superclass is present.
2. **NORM**: Assigned if Normal (NORM) is sole diagnostic superclass.
3. **OTHER_ABNORMAL**: Assigned if any other superclass (STTC, CD, HYP) is present.

### Patient-Wise Split & Full Dataset Training
- **Train**: Folds 1-8 (16,303 records)
- **Validation**: Fold 9 (2,034 records - used strictly for setting selection)
- **Test**: Fold 10 (2,050 records - evaluated ONCE)

Unit tests (pytest) enforce zero overlap of patient_id or ecg_id across train, validation, and test splits. All 7 unit tests run and pass in a fresh clone.

---

## Setup & Execution Guide

### 1. Installation
`ash
git clone -b merge-ecg https://github.com/SiddhiDeshmukh310/CardioScan.git
cd CardioScan
pip install -r requirements.txt
`

### 2. Run Split Integrity Unit Tests (All 7 Tests Pass in Fresh Clone)
`ash
pytest
`

### 3. Evaluate 1D Waveform Model (Full Fold 10 Test Set)
`ash
python src/evaluate_waveform.py
`

### 4. Launch Web Application Demo
`ash
python app/app.py
`
Navigate to http://127.0.0.1:5000 in your web browser.

---

## Limitations & Scope

- **Image Model Limitations**: 2D scanned ECG plot images failed to generalize under leak-free patient splits.
- **Sample Scope**: The web demo features PTB-XL sample records for rapid demonstration.
- **Screening Demo**: CardioScan is designed as an educational screening proof-of-concept.

---

## Citation & Credits

- Dataset provided by **PhysioNet**:
  > Wagner, P., Strodthoff, N., Bousseljot, R. D., Samek, W., & Schaeffter, T. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7(1), 154. 
  > PhysioNet Archive: https://physionet.org/content/ptb-xl/1.0.3/
