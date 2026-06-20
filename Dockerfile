# YT Channel Creator — RunPod Serverless Voice-Worker (OmniVoice TTS + Demucs)
# Build & push:  docker build -t <user>/ytcc-voice:latest .  &&  docker push <user>/ytcc-voice:latest
# Dann auf RunPod als Serverless-Endpoint mit GPU (>=16 GB, z. B. A4000/4090) deployen.

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.8) — von OmniVoice empfohlen
RUN pip3 install --no-cache-dir torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

# OmniVoice (TTS), Demucs (Stem-Trennung), RunPod SDK
RUN pip3 install --no-cache-dir omnivoice demucs runpod requests

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
