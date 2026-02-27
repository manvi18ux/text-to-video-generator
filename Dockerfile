FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "huggingface_hub==0.20.3"
RUN pip install --no-cache-dir "gradio>=4.44.0,<5.0.0"

COPY . .

CMD ["python", "app.py"]