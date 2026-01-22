# WhisperX Modal Transcription Service (FastAPI + Callback)

This project deploys a **Modal** service that:
- Accepts an `audio_url` and a `callback_url` via HTTP `POST`
- Transcribes the audio using **WhisperX (large-v2)** on GPU
- Sends the transcription result to the provided `callback_url`
- Provides a simple `GET /health` endpoint

> The transcription job is executed in the background using `spawn()` and the API returns **202 Accepted** immediately.

---

## Features

- ✅ GPU transcription with WhisperX (`large-v2`)
- ✅ Segment-level timestamps (no word-level alignment)
- ✅ Fire-and-forget background processing (`spawn`)
- ✅ Callback delivery (success/failure payload)
- ✅ Health check endpoint

---

## Requirements

- Python **3.11**
- A Modal account + tokens:
  - `MODAL_TOKEN_ID`
  - `MODAL_TOKEN_SECRET`

---

## Project Structure

Typical setup:
```
.
├─ main.py
├─ pyproject.toml
└─ README.md
```

---

## Install (Local)

> Modal imports your module locally during deploy, so local dependencies must be installed.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

---

## Configure Modal Auth

Set env vars locally:
```bash
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."
```

Or follow Modal's auth instructions (recommended for local dev):
```bash
modal token new
```

---

## Deploy to Modal
```bash
modal deploy -m main
```

This will deploy:

* `POST /transcribe_audio` (FastAPI endpoint)
* `GET /health_check` (FastAPI endpoint)

Modal will output the public endpoint URLs after deployment.

---

## API

### 1) Start transcription (async)

**POST** endpoint (Modal FastAPI endpoint):
```http
POST /transcribe_audio
Content-Type: application/json
```

Body:
```json
{
  "audio_url": "https://example.com/audio.mp3",
  "callback_url": "https://your-server.com/webhook/transcription"
}
```

Response: **202 Accepted**
```json
{
  "status": "accepted",
  "message": "Audio transcription request accepted and processing started",
  "request_id": "b4a5a3b9-9d89-4e7e-8fb1-3f9c6f5b2a6a"
}
```

### 2) Callback payload

Your `callback_url` will receive a `POST` with this JSON payload.

#### Success
```json
{
  "request_id": "b4a5a3b9-9d89-4e7e-8fb1-3f9c6f5b2a6a",
  "status": "completed",
  "message": "Audio transcription completed successfully",
  "timestamp": "2026-01-21T23:59:59.123456",
  "result": {
    "language": "es",
    "segments": [
      {
        "start": 0.0,
        "end": 4.2,
        "text": "Hola, este es un ejemplo..."
      }
    ],
    "duration": 123.45
  },
  "error": null
}
```

#### Failure
```json
{
  "request_id": "b4a5a3b9-9d89-4e7e-8fb1-3f9c6f5b2a6a",
  "status": "failed",
  "message": "Audio transcription failed",
  "timestamp": "2026-01-21T23:59:59.123456",
  "result": null,
  "error": "Some error message"
}
```

---

## Health Check
```http
GET /health_check
```

Response:
```json
{
  "status": "healthy",
  "service": "whisper-transcription"
}
```

---

## Notes / Gotchas

### 1) `ModuleNotFoundError: No module named 'httpx'`

Modal imports your `main.py` locally during `modal deploy`, so you must have local deps installed:
```bash
pip install -e .
```

Alternatively, avoid top-level imports and import dependencies inside Modal-executed functions, but the recommended approach is to keep local deps installed.

### 2) Large audio files (30–100 minutes)

* Ensure your `audio_url` supports direct download (no auth/cookies required).
* Keep an eye on timeout limits:

  * `@app.cls(... timeout=60 * 10 ...)` currently sets 10 minutes.
  * Increase if your 100-minute files take longer.

### 3) WhisperX alignment

This implementation **does not run word-level alignment** (faster, cheaper).
If you need word timestamps, you'll need to load an alignment model and run `whisperx.align(...)` (adds compute time).

---

## CI/CD (GitHub Actions)

Example workflow deploys on push to `main`:

* Uses `pyproject.toml` to install deps
* Runs `modal deploy -m main`
* Requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` secrets

See `.github/workflows/<your-file>.yml`.

---

## License

MIT (adjust as needed).