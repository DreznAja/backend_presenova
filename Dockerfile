FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV + DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python deps (cached layer)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create faces directory
RUN mkdir -p faces

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
