# ================= Stage 1: Build React Frontend =================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ================= Stage 2: Production Python API =================
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV (Mesa GL and GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY ecg_analysis.py ./

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose FastAPI production port
EXPOSE 8000

# Set Python Path and start unified FastAPI production server
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
