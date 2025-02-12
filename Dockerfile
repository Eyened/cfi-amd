#FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# required for opencv
RUN apt update && apt install -y ffmpeg libsm6 libxext6

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY models /app/models
COPY src /app/src

ENTRYPOINT ["python3", "/app/src/main.py"]