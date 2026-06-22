# ===========================================================================
# Dockerfile fuer den RunPod-Audio-Worker
#  - Source Separation (Bandit-v2) + Forced-Alignment (WhisperX)
# ===========================================================================
#
# Basis: RunPod-PyTorch-Image mit CUDA 11.8 (passt zu Bandit-v2: cu118 / torch 2.0).
# Alternativ koennte nvidia/cuda:11.8.0-cudnn8-devel + manuelles Python genutzt
# werden; das RunPod-Image spart Setup.
#
# VERIFY: Tag ggf. an verfuegbare RunPod-Images anpassen (Docker Hub: runpod/pytorch).
# ---------------------------------------------------------------------------
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

# --- System-Pakete: ffmpeg (Konvertierung), git, wget ----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ---------------------------------------------------------------------------
# 1) Bandit-v2 klonen
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/kwatcharasupat/bandit-v2.git /opt/bandit-v2

# Bandit-v2-Requirements installieren, aber Netflix-interne / Trainings-Pakete
# herausfiltern (nflx-*, ray[...]), die fuer reine Inferenz nicht noetig und
# nicht oeffentlich installierbar sind.
# VERIFY: Filterliste gegen die reale requirements.txt des Repos pruefen.
RUN sed -E '/nflx|metaflow|^ray|jasper|manta/d' /opt/bandit-v2/requirements.txt \
        > /opt/bandit-v2/requirements.inference.txt || true \
    && pip install --no-cache-dir -r /opt/bandit-v2/requirements.inference.txt || \
       echo "WARN: Teil-Install der Bandit-Requirements fehlgeschlagen - manuell pruefen"

# ---------------------------------------------------------------------------
# 2) Zenodo-Checkpoint laden
#    Quelle: https://zenodo.org/records/12701995
# ---------------------------------------------------------------------------
# VERIFY: Exakten Datei-Download-Link + Dateinamen aus der Zenodo-Record-Seite
#         eintragen. Zenodo-Direktlinks haben die Form:
#         https://zenodo.org/records/12701995/files/<DATEINAME>?download=1
# VERIFY: BANDIT_CKPT_PATH (in handler.py / unten als ENV) auf den realen
#         entpackten .ckpt-Pfad setzen. Falls das Archiv ein .zip/.tar ist,
#         hier zusaetzlich entpacken.
RUN mkdir -p /opt/checkpoints/bandit-v2
# Mehrsprachiges Modell (deckt DE/EN/… ab). Für rein deutsch ginge auch checkpoint-deu.ckpt.
# Über das Build-ARG BANDIT_CKPT_FILE umstellbar.
ARG BANDIT_CKPT_FILE=checkpoint-multi.ckpt
RUN wget --tries=3 -O /opt/checkpoints/bandit-v2/${BANDIT_CKPT_FILE} \
        "https://zenodo.org/records/12701995/files/${BANDIT_CKPT_FILE}?download=1"

# ---------------------------------------------------------------------------
# 3) Worker-Code + Python-Abhaengigkeiten
# ---------------------------------------------------------------------------
#
# WICHTIG (Versionskonflikt-Fix "torchaudio has no attribute 'AudioMetaData'"):
# torch/torchaudio werden ZUERST und explizit aus dem cu118-Wheel-Index in der
# konsistenten Minor-Version 2.1.2 installiert. torchaudio 2.1.2 besitzt das
# Top-Level torchaudio.AudioMetaData + list_audio_backends, die pyannote.audio
# 3.3.2 (von WhisperX 3.3.1 gepinnt) noch nutzt. So passt der CUDA-Stack zur
# CUDA-11.8-Basis und der AudioMetaData-Importfehler tritt nicht auf.
# Reihenfolge ist bewusst: erst der Torch-Stack, dann die uebrigen Requirements.
RUN pip install --no-cache-dir \
        torch==2.1.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /opt/runpod-audio/requirements.txt
# Restliche Abhaengigkeiten installieren. torch/torchaudio sind bereits in 2.1.2
# (cu118) vorhanden und in requirements.txt identisch gepinnt -> pip laesst sie
# stehen und zieht KEINE CPU-/Default-Wheels nach. Die WhisperX-Transitiv-Deps
# (pyannote.audio 3.3.2, faster-whisper 1.1.0, ctranslate2 3.24.0) sind dort
# kompatibel mitgepinnt.
RUN pip install --no-cache-dir -r /opt/runpod-audio/requirements.txt

COPY handler.py /opt/runpod-audio/handler.py

# ---------------------------------------------------------------------------
# 4) Optional: WhisperX-Modell beim Build vorladen (vermeidet Kaltstart-Download)
# ---------------------------------------------------------------------------
# VERIFY: Modellgroesse muss zu WHISPERX_MODEL passen. large-v2 ist gross (~3 GB);
#         fuer schnellere Kaltstarts ggf. "medium" oder "small" waehlen.
# RUN python -c "import whisperx; whisperx.load_model('large-v2', 'cpu', compute_type='int8')" || \
#     echo "WARN: WhisperX-Preload uebersprungen"

# ---------------------------------------------------------------------------
# ENV-Defaults (im RunPod-Endpoint ueberschreibbar)
# ---------------------------------------------------------------------------
ENV BANDIT_REPO_DIR=/opt/bandit-v2 \
    BANDIT_CKPT_PATH=/opt/checkpoints/bandit-v2/checkpoint-multi.ckpt \
    BANDIT_MODEL_VARIANT=dnr3 \
    BANDIT_FS=44100 \
    WHISPERX_MODEL=large-v2 \
    WHISPERX_COMPUTE_TYPE=float16

WORKDIR /opt/runpod-audio

# RunPod-Serverless startet den Handler-Loop.
CMD ["python", "-u", "handler.py"]
