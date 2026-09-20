# Production Dockerfile for CardioScan Flask Application (Untested)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, model, splits, and application assets
COPY app/ ./app/
COPY src/ ./src/
COPY model/ ./model/
COPY splits/ ./splits/

# Expose Flask default port
EXPOSE 5000

ENV PYTHONPATH=/app

# Start Flask application server
CMD [" python\, \app/app.py\]
