FROM python:3.11-slim

LABEL maintainer="Solar Forecast Pro" \
      description="Physics-based + AI hybrid PV production forecasting"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies for scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libhdf5-dev \
        libnetcdf-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Application code
COPY . .

# Data directory for SQLite
RUN mkdir -p /app/data /app/.cache /app/models

EXPOSE 8501 8000

# Default: run the Streamlit dashboard
CMD ["python", "run.py"]
