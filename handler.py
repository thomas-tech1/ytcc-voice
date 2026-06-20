"""
YT Channel Creator — RunPod Serverless Voice-Worker.

Stellt drei Aktionen bereit (event["input"]["action"]):

1) "tts" / "preview" — OmniVoice Text-to-Speech
   Modi:
     - "clone"  : ref_audio_url (+ optional ref_text)  → Voice-Cloning
     - "design" : instruct (z. B. "female, low pitch, british accent") → Voice-Design
     - "auto"   : ohne Vorgabe → zufällige Stimme
   Weitere Felder: text (Pflicht), speed, duration, language_name (nur Doku).
   Rückgabe: { "audio_b64": <wav base64, 24kHz>, "sr": 24000 }

2) "separate" — Demucs Source-Separation (Stimme vs. Rest)
   Felder: audio_url ODER audio_b64
   Rückgabe: { "music_b64": <no_vocals wav>, "vocals_b64": <vocals wav> }
   → "music_b64" = Musik/Ambience-Stem (Original-Stimme entfernt)

Antworten sind base64-WAV, damit kein zusätzlicher Storage nötig ist.
"""
import os, io, base64, tempfile, subprocess, traceback
import requests
import numpy as np
import torch
import torchaudio
import runpod

HANDLER_VERSION = "numpy-safe-v2"
print(f"[ytcc-voice] handler version: {HANDLER_VERSION}", flush=True)

_OV = None


def _omnivoice():
    """OmniVoice einmalig laden (Cold Start)."""
    global _OV
    if _OV is None:
        from omnivoice import OmniVoice
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        _OV = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dev,
                                        dtype=torch.float16 if dev.startswith("cuda") else torch.float32)
    return _OV


def _dl(url, timeout=180):
    r = requests.get(url, timeout=timeout); r.raise_for_status(); return r.content


def _wav_b64(audio, sr=24000):
    """Robust: akzeptiert torch.Tensor ODER numpy.ndarray, 1-D oder 2-D, und schreibt WAV."""
    t = audio
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    elif not torch.is_tensor(t):
        t = torch.as_tensor(np.asarray(t))
    t = t.detach().to("cpu").float()
    if t.dim() == 1:
        t = t.unsqueeze(0)          # (T,) -> (1, T)
    elif t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    buf = io.BytesIO()
    torchaudio.save(buf, t, sr, format="wav")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tts(inp):
    model = _omnivoice()
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "text fehlt"}
    kw = {}
    if inp.get("speed"):    kw["speed"] = float(inp["speed"])
    if inp.get("duration"): kw["duration"] = float(inp["duration"])
    if inp.get("num_step"): kw["num_step"] = int(inp["num_step"])
    mode = (inp.get("mode") or "auto").lower()
    if mode == "clone" and inp.get("ref_audio_url"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(_dl(inp["ref_audio_url"])); refp = f.name
        audio = model.generate(text=text, ref_audio=refp, ref_text=(inp.get("ref_text") or None), **kw)
    elif mode == "design" and (inp.get("instruct") or "").strip():
        audio = model.generate(text=text, instruct=inp["instruct"].strip(), **kw)
    else:
        audio = model.generate(text=text, **kw)
    first = audio[0] if isinstance(audio, (list, tuple)) else audio
    return {"audio_b64": _wav_b64(first, 24000), "sr": 24000}


def _separate(inp):
    raw = _dl(inp["audio_url"]) if inp.get("audio_url") else base64.b64decode(inp["audio_b64"])
    with tempfile.TemporaryDirectory() as d:
        rawp = os.path.join(d, "in_raw")
        srcp = os.path.join(d, "in.wav")
        with open(rawp, "wb") as f:
            f.write(raw)
        # in echtes WAV normalisieren (mp4/m4a/… → wav)
        subprocess.run(["ffmpeg", "-y", "-i", rawp, "-ac", "2", "-ar", "44100", srcp], capture_output=True)
        if not os.path.exists(srcp):
            return {"error": "ffmpeg konnte das Audio nicht lesen"}
        # Demucs: zwei Stems (Stimme / Rest)
        subprocess.run(["python3", "-m", "demucs", "--two-stems", "vocals", "-o", d, srcp],
                       capture_output=True, check=True)
        base = os.path.join(d, "htdemucs", "in")
        out = {}
        mus = os.path.join(base, "no_vocals.wav")
        voc = os.path.join(base, "vocals.wav")
        if os.path.exists(mus):
            out["music_b64"] = base64.b64encode(open(mus, "rb").read()).decode("ascii")
        if os.path.exists(voc):
            out["vocals_b64"] = base64.b64encode(open(voc, "rb").read()).decode("ascii")
        if not out:
            return {"error": "Demucs lieferte keine Stems"}
        return out


def handler(event):
    inp = (event or {}).get("input") or {}
    action = (inp.get("action") or "tts").lower()
    try:
        if action in ("tts", "preview"):
            return _tts(inp)
        if action == "separate":
            return _separate(inp)
        return {"error": f"unbekannte action: {action}"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[:1500]}


runpod.serverless.start({"handler": handler})
