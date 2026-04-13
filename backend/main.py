import io
import os
from pathlib import Path

import librosa
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import median_filter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SOUND_DIR = Path(__file__).resolve().parent.parent / "sound"


@app.get("/api/sounds")
def list_sounds():
    extensions = {".mp3", ".wav", ".flac", ".ogg"}
    files = [
        f.name for f in SOUND_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return {"files": sorted(files)}


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    path = SOUND_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "File not found")
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    return FileResponse(path, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/api/metadata/{filename}")
def get_metadata(filename: str):
    path = SOUND_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "File not found")
    duration = librosa.get_duration(path=str(path))
    return {"duration": duration, "filename": filename}


@app.get("/api/spectrogram/{filename}")
def get_spectrogram(filename: str):
    path = SOUND_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "File not found")

    y, sr = librosa.load(str(path), sr=None, mono=True)

    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # --- Adaptive spectral gating ---
    # Per-frequency noise estimate from the quieter portions of the signal
    noise_floor = np.percentile(S, 25, axis=1, keepdims=True)
    # Soft mask: ratio of signal above noise vs total
    S_gate = np.maximum(S - noise_floor * 1.5, 0)
    mask = S_gate / (S + 1e-10)
    # Smooth the mask to avoid harsh edges
    mask = median_filter(mask, size=(3, 5))
    S_clean = S * mask

    # Remove speckle noise with a small median filter
    S_clean = median_filter(S_clean, size=(3, 3))

    S_db = librosa.amplitude_to_db(S_clean, ref=np.max)

    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    S_norm = np.power(S_norm, 0.55)

    p_hi = np.percentile(S_norm, 99.5)
    S_norm = np.clip(S_norm / (p_hi + 1e-8), 0, 1)

    # Adaptive soft threshold derived from signal statistics
    # Use the median of non-zero values as a proxy for the residual noise level
    nz = S_norm[S_norm > 0]
    noise_thresh = np.median(nz) + 0.25 * np.std(nz) if len(nz) > 0 else 0.05
    noise_thresh = np.clip(noise_thresh, 0.03, 0.35)
    S_norm = np.where(
        S_norm < noise_thresh, 0,
        (S_norm - noise_thresh) / (1 - noise_thresh)
    )

    S_norm = S_norm[::-1, :]

    rgba = cm.magma(S_norm)
    pixels = (rgba[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(pixels, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
