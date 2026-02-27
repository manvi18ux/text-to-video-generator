FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "torch" \
    "diffusers" \
    "transformers" \
    "accelerate" \
    "opencv-python" \
    "numpy" \
    "imageio" \
    "imageio-ffmpeg" \
    "Pillow" \
    "gradio==5.50.0"

COPY . .

CMD ["python", "app.py"]