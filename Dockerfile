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
    "huggingface_hub==0.25.0" \
    "diffusers" \
    "transformers" \
    "accelerate" \
    "torch" \
    "opencv-python" \
    "numpy" \
    "imageio" \
    "imageio-ffmpeg" \
    "Pillow" \
    "gradio==4.44.1"

COPY . .

CMD ["python", "app.py"]