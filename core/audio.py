import os
from dataclasses import dataclass

from core.interface import _groq_client

WHISPER_MODEL = "whisper-large-v3-turbo"
WHISPER_PRICE_PER_HOUR = 0.04
MIN_BILLED_SECONDS = 10
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass
class Transcription:
    text: str
    duration_s: float
    cost_usd: float
    model: str


def transcribe_audio(path: str) -> Transcription:
    if os.path.getsize(path) > MAX_UPLOAD_BYTES:
        raise ValueError("Audio file is over 25 MB. Use a shorter clip.")

    with open(path, "rb") as f:
        result = _groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model=WHISPER_MODEL,
            response_format="verbose_json",
        )

    text = (result.text or "").strip()
    if not text:
        raise ValueError("No speech detected in the audio.")

    duration = float(getattr(result, "duration", 0) or 0)
    cost = max(duration, MIN_BILLED_SECONDS) / 3600 * WHISPER_PRICE_PER_HOUR
    return Transcription(text=text, duration_s=duration, cost_usd=cost, model=WHISPER_MODEL)