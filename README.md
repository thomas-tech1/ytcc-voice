# YTCC Voice-Worker (RunPod Serverless)

Eigene „lokale AI" für Schritt 5 (Voiceover): **OmniVoice** (TTS, Cloning, Voice-Design) + **Demucs** (Stimme/Musik trennen).

## Voraussetzungen (einmalig)

- **Docker Desktop** auf deinem Mac installieren: https://www.docker.com/products/docker-desktop/ → starten (Wal-Symbol oben in der Menüleiste muss laufen).
- **Docker-Hub-Account** (kostenlos): https://hub.docker.com/ → Benutzername merken (steht unten überall für `<dockerhub-user>`).

## Deploy

1. **Image bauen & pushen — lokal im Terminal auf deinem Mac**
   Terminal öffnen, in DIESEN Ordner wechseln und einloggen:
   ```bash
   cd "/Users/thomasleibel/Claude/Projects/Fellnasenwissen/ytchannelcreator-app/runpod-voice"
   docker login        # einmalig: Docker-Hub-User + Passwort
   ```
   Dann bauen & pushen. **Wichtig:** RunPod-GPUs sind `amd64` — auf einem Apple-Silicon-Mac (M1–M4) MUSS `--platform linux/amd64` mit dabei sein, sonst startet der Worker auf RunPod nicht:
   ```bash
   docker buildx build --platform linux/amd64 \
       -t <dockerhub-user>/ytcc-voice:latest --push .
   ```
   (Das `--push` lädt direkt zu Docker Hub hoch — ein separates `docker push` ist dann nicht nötig.)

   **Alternative ohne lokales Docker:** Du kannst diesen Ordner in ein GitHub-Repo legen und RunPod das Image direkt aus dem Repo bauen lassen (RunPod → Serverless → New Endpoint → „Import Git Repository"). Dann brauchst du Docker lokal gar nicht.

2. **RunPod → Serverless → New Endpoint**
   - Container Image: `<dockerhub-user>/ytcc-voice:latest`
   - GPU: ≥ 16 GB (A4000 / 4090 reicht)
   - **Network Volume** anhängen und auf `/runpod-volume` mounten → cached die OmniVoice-Modelle (HF), sonst lädt jeder Cold Start neu.
   - Container Disk: ≥ 20 GB.
3. **Endpoint-ID + API-Key** in die `secrets.env` der App eintragen:
   ```
   YTCC_RUNPOD_API_KEY=...
   YTCC_RUNPOD_VOICE_ENDPOINT=<endpoint-id>
   ```

## API (event["input"])

**TTS / Preview**
```json
{ "action": "tts", "text": "Hallo!", "mode": "design",
  "instruct": "male, warm, british accent", "speed": 1.0 }
```
- `mode`: `clone` (+ `ref_audio_url`, optional `ref_text`) · `design` (+ `instruct`) · `auto`
- Rückgabe: `{ "audio_b64": "<wav 24kHz base64>", "sr": 24000 }`

**Stem-Trennung**
```json
{ "action": "separate", "audio_url": "https://…/clip.mp4" }
```
- Rückgabe: `{ "music_b64": "<no_vocals.wav>", "vocals_b64": "<vocals.wav>" }`
- `music_b64` = Musik-/Ambience-Stem (Original-Seedance-Stimme entfernt).

## Hinweise
- Erster Aufruf (Cold Start) lädt das OmniVoice-Modell von HuggingFace — mit Network Volume nur einmal.
- Voice-Design-Attribute (Auszug): gender (male/female), age (child…elderly), pitch (very low…very high), accent (American/British…), style (whisper). Kommagetrennt kombinierbar.
- Cloning braucht 3–25 Sek. saubere Referenz-Audio.
