"""
RunPod-Serverless-Worker fuer Audio-Verarbeitung (YTChannelCreator / Fellnasenwissen).

Zwei GPU-Funktionen in einem Worker:

  1) "separate" : Cinematic Source Separation in 3 Stems (SPEECH / MUSIC / EFFECTS)
                  via Bandit-v2 (Karn Watcharasupat, Apache-2.0).
                  Repo:   https://github.com/kwatcharasupat/bandit-v2
                  Weights: https://zenodo.org/records/12701995

  2) "align"    : Forced-Alignment / ASR-Timing (Segment- + Wort-Zeitstempel)
                  via WhisperX (https://github.com/m-bain/whisperX).

  3) "both"     : Erst separieren, dann den SPEECH-Stem alignen.

Handler-API (RunPod):
    event["input"] = {
        "mode": "separate" | "align" | "both",
        "audio_url":    "<http(s) url>",      # ODER ...
        "audio_base64": "<base64>",           # ... eines von beiden Pflicht
        "lang": "de"                          # optional, nur fuer align (Default: Auto-Detect)
    }

Rueckgabe (JSON-serialisierbar):
    separate -> {"ok": true, "stems": {"speech": "<b64 wav>", "music": "<b64 wav>", "effects": "<b64 wav>"}}
    align    -> {"ok": true, "segments": [{start,end,text, words:[{word,start,end}]}]}
    both     -> beide Keys
    Fehler   -> {"ok": false, "error": "..."}

WICHTIG: Alle Stellen, die beim echten Deploy gegen die reale Bandit-v2-API
geprueft werden muessen, sind mit "# VERIFY:" markiert.
"""

import os
import base64
import shutil
import tempfile
import subprocess
import traceback

import requests

import runpod


# ---------------------------------------------------------------------------
# Konfiguration ueber ENV-Variablen (im Docker-Image / RunPod-Endpoint setzbar)
# ---------------------------------------------------------------------------

# Pfad zum geklonten Bandit-v2-Repo im Container.
BANDIT_REPO_DIR = os.environ.get("BANDIT_REPO_DIR", "/opt/bandit-v2")

# Pfad zum heruntergeladenen Zenodo-Checkpoint (.ckpt) im Container.
# VERIFY: Exakten Dateinamen aus https://zenodo.org/records/12701995 eintragen
#         (im Dockerfile per wget gezogen). Platzhalter unten.
BANDIT_CKPT_PATH = os.environ.get(
    "BANDIT_CKPT_PATH", "/opt/checkpoints/bandit-v2/checkpoint.ckpt"
)

# Bandit-Modellvariante. Bestimmt u.a. die Stem-Namen, die das Modell ausgibt.
# VERIFY: Korrekten Variantennamen laut configs/ bzw. expt/ des Repos setzen,
#         z.B. "dnr3" / "musvdb" o.ae. Muss zum Checkpoint passen.
BANDIT_MODEL_VARIANT = os.environ.get("BANDIT_MODEL_VARIANT", "dnr3")

# Sample-Rate, mit der Bandit-v2 arbeitet (DnR ist i.d.R. 44100 Hz).
# VERIFY: Mit der Inferenz-Config (expt/inference.yaml -> fs) abgleichen.
BANDIT_FS = int(os.environ.get("BANDIT_FS", "44100"))

# WhisperX-Modellgroesse (tiny/base/small/medium/large-v2/large-v3).
WHISPERX_MODEL = os.environ.get("WHISPERX_MODEL", "large-v2")

# Compute-Type fuer faster-whisper (float16 auf GPU, int8 als Fallback).
WHISPERX_COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16")

DEVICE = "cuda"

# Mapping der internen Bandit-Stem-Namen auf unsere API-Keys.
# VERIFY: Die tatsaechlichen Schluessel in output["estimates"] des Modells
#         (siehe separate_stems) koennen abweichen, z.B. "dialog"/"sfx".
#         Hier ggf. anpassen.
STEM_NAME_MAP = {
    "speech": ["speech", "dialog", "dialogue", "vocals", "vox"],
    "music": ["music", "mus"],
    "effects": ["effects", "effect", "sfx", "fx", "sound_effects"],
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Laden / Konvertieren / Kodieren
# ---------------------------------------------------------------------------

def _download_to_file(audio_url: str, dest_path: str) -> None:
    """Laedt eine http(s)-URL als Stream in eine Datei."""
    with requests.get(audio_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB Chunks
                if chunk:
                    fh.write(chunk)


def _b64_to_file(audio_b64: str, dest_path: str) -> None:
    """Dekodiert einen base64-String in eine Datei."""
    # Erlaubt sowohl reines base64 als auch data-URIs (data:audio/wav;base64,...).
    if "," in audio_b64 and audio_b64.strip().lower().startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]
    with open(dest_path, "wb") as fh:
        fh.write(base64.b64decode(audio_b64))


def _load_input_audio(inp: dict, dest_path: str) -> None:
    """
    Holt das Eingabe-Audio aus event["input"] (url ODER base64) in dest_path.
    Wirft ValueError, wenn keine Quelle angegeben ist.
    """
    audio_url = inp.get("audio_url")
    audio_b64 = inp.get("audio_base64")

    if audio_url:
        _download_to_file(audio_url, dest_path)
    elif audio_b64:
        _b64_to_file(audio_b64, dest_path)
    else:
        raise ValueError("Weder 'audio_url' noch 'audio_base64' angegeben.")


def _ffmpeg_convert(src_path: str, dst_path: str, sample_rate: int, channels: int) -> None:
    """
    Konvertiert beliebiges Eingabe-Audio via ffmpeg in ein WAV (PCM s16le)
    mit gewuenschter Sample-Rate und Kanalzahl.

      - separate: 44.1k Stereo (channels=2)
      - align:    16k Mono    (channels=1)
    """
    cmd = [
        "ffmpeg",
        "-y",                       # ueberschreiben ohne Rueckfrage
        "-i", src_path,
        "-ar", str(sample_rate),    # Sample-Rate
        "-ac", str(channels),       # Kanalzahl
        "-c:a", "pcm_s16le",        # unkomprimiertes WAV
        "-f", "wav",
        dst_path,
    ]
    # ffmpeg gibt seine Logs auf stderr aus; bei Fehler werfen wir mit Kontext.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg-Konvertierung fehlgeschlagen (rc={proc.returncode}): "
            f"{proc.stderr[-2000:]}"
        )


def _file_to_b64(path: str) -> str:
    """Liest eine Datei und gibt sie als base64-String zurueck."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _match_stem_key(internal_name: str) -> str | None:
    """
    Bildet einen vom Modell gelieferten Stem-Namen auf unsere API-Keys
    (speech/music/effects) ab. Gibt None zurueck, wenn nicht zuordenbar.
    """
    low = internal_name.strip().lower()
    for api_key, aliases in STEM_NAME_MAP.items():
        if low == api_key or low in aliases:
            return api_key
    return None


# ---------------------------------------------------------------------------
# 1) Source Separation: Bandit-v2
# ---------------------------------------------------------------------------

def separate_stems(in_wav: str, out_dir: str) -> dict:
    """
    Trennt in_wav (44.1k Stereo WAV) in 3 Stems und schreibt sie nach out_dir.

    Gibt ein Dict zurueck:
        {"speech": "<pfad>.wav", "music": "<pfad>.wav", "effects": "<pfad>.wav"}

    -----------------------------------------------------------------------
    VERIFY: Dies ist die zentrale Stelle, die gegen die reale Bandit-v2-API
            verifiziert werden muss. Laut Repo-`inference.py` laeuft die
            Inferenz ueber Hydra + ein PyTorch-Lightning-"System":

                from src.system.utils import build_system
                system = build_system(cfg)            # cfg aus expt/inference.yaml
                system.load_state_dict(
                    torch.load(ckpt_path)["state_dict"], strict=False)
                system.to("cuda"); system.eval()
                with torch.inference_mode():
                    output = system.inference_handler(audio[None].cuda(), system.model)
                # output["estimates"][stem]["audio"][0]  -> (channels, samples)

            Zwei robuste Deploy-Optionen, je nachdem was im Container leichter
            stabil laeuft:

            (A) CLI-Aufruf von inference.py (empfohlen, da Hydra-Config-getrieben):
                  python inference.py \
                    ckpt_path=<BANDIT_CKPT_PATH> \
                    test_audio=<in_wav> \
                    output_path=<out_dir> \
                    model_variant=<BANDIT_MODEL_VARIANT> \
                    fs=<BANDIT_FS>
                Ergebnis-Dateien heissen dann "<stem>_estimate.wav" in out_dir.
                # VERIFY: exakte Hydra-Override-Keys (config_path="expt",
                #         config_name="inference") + ob test_audio/output_path
                #         in der Default-Config existieren oder ergaenzt werden
                #         muessen.

            (B) Python-In-Process-Aufruf (schneller, kein Subprozess-Overhead),
                aber erfordert korrekt zusammengebaute Hydra-DictConfig.

            Unten ist Option (A) implementiert (Subprozess), da sie der
            dokumentierten inference.py am naechsten kommt. Die In-Process-
            Variante ist als Kommentar skizziert.
    -----------------------------------------------------------------------
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Option (A): inference.py als Subprozess -------------------------
    # VERIFY: Override-Keys/-Werte gegen expt/inference.yaml & configs/ pruefen.
    cmd = [
        "python", "inference.py",
        f"ckpt_path={BANDIT_CKPT_PATH}",
        f"test_audio={in_wav}",
        f"output_path={out_dir}",
        f"model_variant={BANDIT_MODEL_VARIANT}",
        f"fs={BANDIT_FS}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=BANDIT_REPO_DIR,     # inference.py erwartet das Repo als CWD (configs/expt relativ)
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Bandit-v2 Inferenz fehlgeschlagen "
            f"(rc={proc.returncode}).\nSTDOUT:\n{proc.stdout[-2000:]}\n"
            f"STDERR:\n{proc.stderr[-2000:]}"
        )

    # --- Ergebnis-Dateien einsammeln und auf API-Keys mappen -------------
    # inference.py schreibt "<stem>_estimate.wav". Wir suchen sie ein.
    found = {}
    for fname in os.listdir(out_dir):
        if not fname.lower().endswith(".wav"):
            continue
        # "speech_estimate.wav" -> "speech"
        stem_raw = fname.rsplit(".", 1)[0]
        if stem_raw.endswith("_estimate"):
            stem_raw = stem_raw[: -len("_estimate")]
        api_key = _match_stem_key(stem_raw)
        if api_key:
            found[api_key] = os.path.join(out_dir, fname)

    # Pflicht: alle drei Stems vorhanden. Andernfalls Mapping pruefen.
    missing = [k for k in ("speech", "music", "effects") if k not in found]
    if missing:
        raise RuntimeError(
            "Nicht alle Stems gefunden. Fehlend: "
            f"{missing}. Gefundene WAVs in {out_dir}: "
            f"{[f for f in os.listdir(out_dir) if f.endswith('.wav')]}. "
            "# VERIFY: STEM_NAME_MAP an die realen Bandit-Stem-Namen anpassen."
        )

    return found

    # --- Option (B): In-Process-Skizze (zur Referenz, NICHT aktiv) -------
    # import sys
    # sys.path.insert(0, BANDIT_REPO_DIR)
    # import torch, torchaudio as ta
    # from omegaconf import OmegaConf
    # from src.system.utils import build_system
    # cfg = OmegaConf.load(os.path.join(BANDIT_REPO_DIR, "expt", "inference.yaml"))
    # cfg.model_variant = BANDIT_MODEL_VARIANT
    # cfg.fs = BANDIT_FS
    # system = build_system(cfg)
    # system.load_state_dict(torch.load(BANDIT_CKPT_PATH)["state_dict"], strict=False)
    # system.to("cuda").eval()
    # audio, fs = ta.load(in_wav)
    # if fs != BANDIT_FS:
    #     audio = ta.functional.resample(audio, fs, BANDIT_FS)
    # with torch.inference_mode():
    #     output = system.inference_handler(audio[None].to("cuda"), system.model)
    # for stem in output["estimates"]:
    #     ta.save(os.path.join(out_dir, f"{stem}_estimate.wav"),
    #             output["estimates"][stem]["audio"][0].cpu(), BANDIT_FS)


# ---------------------------------------------------------------------------
# 2) Forced-Alignment / ASR-Timing: WhisperX
# ---------------------------------------------------------------------------

# Modelle werden lazy geladen und prozessweit gecached (Kaltstart vermeiden).
_WHISPER_MODEL = None
_ALIGN_CACHE = {}  # lang_code -> (align_model, metadata)


def _get_whisper_model():
    """Laedt das WhisperX-ASR-Modell einmalig und cached es."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisperx
        _WHISPER_MODEL = whisperx.load_model(
            WHISPERX_MODEL,
            DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
        )
    return _WHISPER_MODEL


def _get_align_model(lang_code: str):
    """Laedt (und cached) das Alignment-Modell fuer eine Sprache."""
    import whisperx
    if lang_code not in _ALIGN_CACHE:
        model_a, metadata = whisperx.load_align_model(
            language_code=lang_code, device=DEVICE
        )
        _ALIGN_CACHE[lang_code] = (model_a, metadata)
    return _ALIGN_CACHE[lang_code]


def align_audio(in_wav: str, lang: str | None = None) -> list:
    """
    Transkribiert + alignt in_wav (16k Mono WAV) mit WhisperX.

    lang: ISO-Code (z.B. "de"). None => Auto-Detect.

    Rueckgabe: Liste von Segmenten
        [{"start": float, "end": float, "text": str,
          "words": [{"word": str, "start": float, "end": float}, ...]}]
    """
    import whisperx

    model = _get_whisper_model()

    # WhisperX laedt Audio als float32-Numpy bei 16 kHz.
    audio = whisperx.load_audio(in_wav)

    # 1) Transkription (grobe Segment-Zeiten). lang als Hint, sonst Auto-Detect.
    transcribe_kwargs = {"batch_size": 16}
    if lang:
        transcribe_kwargs["language"] = lang
    result = model.transcribe(audio, **transcribe_kwargs)

    detected_lang = result.get("language", lang or "en")

    # 2) Forced-Alignment fuer praezise Wort-Zeitstempel.
    segments_out = []
    try:
        model_a, metadata = _get_align_model(detected_lang)
        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        raw_segments = aligned.get("segments", [])
    except Exception:
        # Falls fuer die Sprache kein Alignment-Modell verfuegbar ist,
        # geben wir wenigstens die groben Transkriptions-Segmente zurueck.
        raw_segments = result.get("segments", [])

    # 3) In unser stabiles Ausgabeformat normalisieren.
    for seg in raw_segments:
        words = []
        for w in seg.get("words", []) or []:
            # WhisperX liefert bei manchen Tokens keine start/end (z.B. Zahlen).
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({
                "word": w.get("word", "").strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
        segments_out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": (seg.get("text") or "").strip(),
            "words": words,
        })

    return segments_out


# ---------------------------------------------------------------------------
# RunPod-Handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    """
    RunPod-Serverless-Einstiegspunkt.

    Erwartet event["input"] mit Feldern mode / audio_url|audio_base64 / lang.
    Gibt immer ein JSON-serialisierbares Dict zurueck.
    """
    workdir = None
    try:
        inp = event.get("input") or {}
        mode = (inp.get("mode") or "both").strip().lower()
        lang = inp.get("lang")  # optional, None = Auto-Detect

        if mode not in ("separate", "align", "both"):
            return {"ok": False, "error": f"Unbekannter mode: {mode!r}"}

        # Eigenes Tempdir pro Request (wird am Ende sauber geloescht).
        workdir = tempfile.mkdtemp(prefix="ytcc_audio_")
        raw_in = os.path.join(workdir, "input_raw")

        # Eingabe-Audio holen (url ODER base64).
        _load_input_audio(inp, raw_in)

        result: dict = {"ok": True}
        speech_wav_for_align = None  # Pfad zum 16k-Mono-WAV des SPEECH-Stems

        # --- Separation -------------------------------------------------
        if mode in ("separate", "both"):
            sep_in = os.path.join(workdir, "sep_input_44k_stereo.wav")
            _ffmpeg_convert(raw_in, sep_in, sample_rate=BANDIT_FS, channels=2)

            sep_out_dir = os.path.join(workdir, "stems")
            stem_paths = separate_stems(sep_in, sep_out_dir)

            # WAV-Stems als base64 zurueckgeben.
            #
            # HINWEIS (Skalierung): base64-WAV ist fuer kurze Folgen voellig ok.
            # Bei langen Audios (mehrere Minuten Stereo @ 44.1k) wird der
            # Payload sehr gross. Dann besser: Stems nach Cloudflare R2 / S3
            # hochladen und nur die URLs zurueckgeben (optionaler Pfad).
            # base64 bleibt hier der Default.
            result["stems"] = {
                "speech": _file_to_b64(stem_paths["speech"]),
                "music": _file_to_b64(stem_paths["music"]),
                "effects": _file_to_b64(stem_paths["effects"]),
            }

            # Fuer mode="both" den SPEECH-Stem als align-Quelle vormerken.
            speech_wav_for_align = stem_paths["speech"]

        # --- Alignment --------------------------------------------------
        if mode in ("align", "both"):
            # Quelle: bei "both" der separierte SPEECH-Stem, sonst das Roh-Input.
            align_source = speech_wav_for_align if speech_wav_for_align else raw_in

            align_in = os.path.join(workdir, "align_input_16k_mono.wav")
            _ffmpeg_convert(align_source, align_in, sample_rate=16000, channels=1)

            result["segments"] = align_audio(align_in, lang=lang)

        return result

    except Exception as exc:  # noqa: BLE001 - bewusst: Fehler als JSON zurueck
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-2000:],
        }
    finally:
        # Sauberes Cleanup des Tempdirs in jedem Fall.
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)


# RunPod-Serverless-Loop starten.
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
