# CardioScan - AI Powered ECG Disease Detection

**Final Year Project 2026**

A web application that analyzes **12-lead ECG images** using computer vision and signal processing to detect heart rate, rhythm abnormalities, and possible cardiac conditions.

## Features
- Grid removal from paper ECG images
- Accurate R-peak detection
- HRV Metrics: BPM, RMSSD, RR Interval Std Dev, CV
- Rule-based classification (Normal Sinus Rhythm, Suspected AFib, Bradycardia, Tachycardia)
- Clean interactive web interface

## Technologies Used
- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV
- **Signal Processing**: SciPy, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Rule-based engine + SVM

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/CardioScan.git
   cd CardioScan
Create virtual environment:Bashpython -m venv venv
venv\Scripts\activate        # On Windows
Install dependencies:Bashpip install -r requirements.txt
Run the application:Bashpython app.py
---



