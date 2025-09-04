# syntax=docker/dockerfile:1
FROM python:3.10-slim

# 1) Minimal system libs for PyMuPDF; no build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Install CPU-only PyTorch FIRST (no CUDA pulls)
#    Pick a stable, smaller torch (2.4.1 is a good sweet spot)
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1

# 3) Then the rest of your deps (no torch here)
#    If you keep requirements.txt, REMOVE torch from it.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 4) App code
COPY batch_carbon_filter.py carbon_filter.py /app/

# 5) Non-root (you can run as root at runtime if you must)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-u", "/app/batch_carbon_filter.py"]
