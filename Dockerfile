# Use official Python 3.9 image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/hf_data/"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download.py .
RUN python3 download.py

# Copy application c    ode
COPY app.py app.py

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "app.py"]
