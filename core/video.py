import os
import subprocess
import tempfile

import imageio_ffmpeg

from core.audio import Transcription, transcribe_audio


def extract_audio(video_path: str) -> str:
    """Pull the sound out of a video as a small mono mp3. Returns the temp file path."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out.close()
    cmd = [ffmpeg, "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", out.name]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0 or os.path.getsize(out.name) == 0:
        os.remove(out.name)
        err = result.stderr or ""
        if "does not contain any stream" in err or "Output file is empty" in err:
            raise ValueError("This video has no sound track, so there is nothing to transcribe.")
        raise ValueError("ffmpeg failed: " + err[-600:])
    return out.name


def transcribe_video(video_path: str) -> Transcription:
    audio_path = extract_audio(video_path)
    try:
        return transcribe_audio(audio_path)
    finally:
        os.remove(audio_path)