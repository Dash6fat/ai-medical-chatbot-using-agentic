import io
import json
import wave
from typing import Any

from config import MAX_VOICE_SECONDS, VOICE_SAMPLE_RATE, VOSK_MODEL_PATH


def load_voice_model() -> Any | None:
    if not VOSK_MODEL_PATH.is_dir():
        return None

    from vosk import Model

    return Model(str(VOSK_MODEL_PATH))


def transcribe_wav(model: Any, audio: bytes) -> str:
    try:
        recording = wave.open(io.BytesIO(audio), "rb")
    except (EOFError, wave.Error) as exc:
        raise ValueError("The recording is not a valid WAV file.") from exc

    with recording:
        if (
            recording.getnchannels() != 1
            or recording.getsampwidth() != 2
            or recording.getframerate() != VOICE_SAMPLE_RATE
            or recording.getcomptype() != "NONE"
        ):
            raise ValueError("Recording must be mono 16 kHz PCM audio.")
        if recording.getnframes() > MAX_VOICE_SECONDS * VOICE_SAMPLE_RATE:
            raise ValueError(f"Voice recordings must be {MAX_VOICE_SECONDS} seconds or shorter.")

        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(model, VOICE_SAMPLE_RATE)
        results = []
        while True:
            chunk = recording.readframes(8000)
            if not chunk:
                break
            if recognizer.AcceptWaveform(chunk):
                results.append(json.loads(recognizer.Result()).get("text", ""))
        results.append(json.loads(recognizer.FinalResult()).get("text", ""))

    return " ".join(text for text in results if text).strip()
