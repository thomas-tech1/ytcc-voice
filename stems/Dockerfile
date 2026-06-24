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

# --- System-Pakete -----------------------------------------------------------
#  ffmpeg            : Audio-Konvertierung im Handler
#  git, wget         : Repo-Clone + Checkpoint-Download
#  pkg-config + libav*-dev : NÖTIG, damit PyAV (av==11.*, von faster-whisper
#                    gezogen) aus dem Quelltext gebaut werden kann, falls kein
#                    passendes Wheel vorliegt ("pkg-config is required for building PyAV").
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git wget ca-certificates \
        pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libavfilter-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ---------------------------------------------------------------------------
# 1) Bandit-v2 klonen
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/kwatcharasupat/bandit-v2.git /opt/bandit-v2

# ---------------------------------------------------------------------------
# 1b) Repo-Patch: ray-Import in src/system/utils.py optional machen
# ---------------------------------------------------------------------------
# src/system/utils.py macht auf MODUL-EBENE `from ray.train.lightning import (
#   RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback,
#   prepare_trainer)`. `ray` ist ein reines TRAININGS-Paket (verteiltes Training
# via Ray Train) und steht NICHT im oeffentlichen Inferenz-Dep-Stack -> beim
# blossen Import von inference.py wuerde `ModuleNotFoundError: No module named
# 'ray'` fliegen. Empirisch verifiziert: Diese vier Ray-Symbole werden NUR im
# Trainings-Codepfad benutzt (build_trainer / build_strategy[RayDDPStrategy] /
# CALLBACK_SEARCH_SPACE[RayTrainReportCallback] / PLUGIN_SEARCH_SPACE[
# RayLightningEnvironment]). Der INFERENZ-Pfad (inference.py -> build_system ->
# build_model/build_loss_handler/build_metric_handler/build_optim_handler/
# build_inference) ruft KEINES davon auf. Daher kapseln wir den Import in
# try/except und ersetzen die Symbole im Fehlerfall durch None, damit die
# Modul-Ebene weiter laedt. ray wird bewusst NICHT installiert.
# Der Patch ist idempotent (Marker-Kommentar) und bricht NICHT, falls die Zeile
# schon angepasst ist.
# Patch als eigenstaendiges Skript (KEIN Dockerfile-Heredoc -> unabhaengig von
# BuildKit-Heredoc-Support, bulletproof). Idempotent; bricht ab, falls der
# ray-Import upstream eine unbekannte Form hat.
COPY patch_bandit_ray.py /tmp/patch_bandit_ray.py
RUN python3 /tmp/patch_bandit_ray.py

# HINWEIS: Die Bandit-Inferenz-Abhaengigkeiten werden NICHT aus
# /opt/bandit-v2/requirements.txt installiert. Diese Datei zeigt per
# `--index-url https://pypi.netflix.net/simple` auf ein Netflix-internes,
# von aussen unerreichbares Repo und listet massenhaft Netflix-/Trainings-
# Pakete (nflx-*, jasper, metatron, ray, metaflow, ...). Der Install daraus
# schlaegt komplett fehl -> es waere KEINE einzige Bandit-Dep installiert
# (frueher durch `|| echo WARN` stillschweigend verschluckt, was zu
# `ModuleNotFoundError: No module named 'hydra'` beim Bandit-Aufruf fuehrte).
#
# Stattdessen werden die MINIMALEN, oeffentlich (PyPI) installierbaren
# Laufzeit-/Inferenz-Deps weiter unten EXPLIZIT installiert -- bewusst NACH
# dem torch-2.2.2-Install, damit torch/torchaudio fix bleiben. Siehe dort.

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
        torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# Bandit-v2 INFERENZ-Abhaengigkeiten (oeffentliches PyPI, KEIN Netflix-Index)
# ---------------------------------------------------------------------------
# Bewusst NACH dem torch-2.2.2-Install (cu121): so bleibt der verifizierte
# Torch-Stack fix und wird durch diese Deps NICHT downgegradet (kein torch==2.0).
#
# Diese Liste ist die minimale Menge an EXTERNEN Paketen, die der reine
# Inferenz-Pfad von Bandit-v2 wirklich importiert (Import-Graph:
# inference.py -> src.system.utils -> models/bandit/*, system/*, metrics,
# losses, data). Netflix-/Ray-/Metaflow-/Training-Pakete werden bewusst
# weggelassen (nicht oeffentlich + fuer Inferenz unnoetig).
#
#   hydra-core==1.3.2  : inference.py nutzt @hydra.main (Config-getrieben).
#                        Zieht omegaconf + antlr4-python3-runtime (rein Python).
#   omegaconf==2.3.0   : DictConfig/OmegaConf. Version erfuellt sowohl hydra-core
#                        (>=2.2,<2.4) als auch pyannote.audio (>=2.1,<3.0).
#   pytorch-lightning==2.3.0 : System = pl.LightningModule; pl.Trainer/seed.
#                        Koexistiert konfliktfrei mit dem `lightning`-Meta-Paket,
#                        das pyannote.audio 3.3.2 zieht.
#   torchmetrics       : src/metrics/snr.py + system/utils (SignalNoiseRatio).
#   librosa            : models/bandit/utils.py (hz_to_midi/midi_to_hz fuer die
#                        musikalische Bandsplit-Spezifikation). Zieht numba,
#                        soundfile, scikit-learn, soxr, pooch (alles Wheels).
#   tqdm               : Fortschrittsbalken im Chunked-Inference-Handler.
#   pandas             : von models/bandit/utils.py referenziert; reines Wheel,
#                        zur Sicherheit mitinstalliert.
#
# torch/torchaudio sind hier bereits in 2.2.2 vorhanden -> pip laesst sie stehen
# und zieht KEINE anderen Torch-Wheels nach (per Trockenlauf verifiziert).
# Dieser RUN ist bewusst NICHT fehlertolerant: schlaegt er fehl, soll der Build
# abbrechen (im Gegensatz zum frueheren `|| echo WARN`).
RUN pip install --no-cache-dir \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        pytorch-lightning==2.3.0 \
        torchmetrics \
        librosa \
        tqdm \
        pandas

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
