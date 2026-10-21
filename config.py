import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = 20
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_VOICE_BYTES = 20 * 1024 * 1024
MAX_VOICE_SECONDS = 600
MAX_DOCUMENT_PAGES = 10
MAX_DOCUMENT_TEXT_CHARS = 8000
VOICE_SAMPLE_RATE = 16000
PERSIST_CHAT_HISTORY = os.getenv("CHATBOT_PERSIST_HISTORY", "true").casefold() in {
    "1",
    "true",
    "yes",
}

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
VOSK_MODEL_PATH = Path(
    os.getenv(
        "VOSK_MODEL_PATH",
        str(Path(__file__).parent / "models" / "vosk-model-small-en-us-0.15"),
    )
).expanduser()


def resolve_tesseract_executable() -> str | None:
    """Return a Tesseract executable path if one is discoverable on this machine."""
    if TESSERACT_CMD:
        candidate = Path(TESSERACT_CMD).expanduser()
        if candidate.is_file():
            return str(candidate)

    env_path = shutil.which("tesseract")
    if env_path:
        return env_path

    candidates: list[Path] = []
    if os.name == "nt":
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]
        candidates.extend(
            Path(base) / "Tesseract-OCR" / "tesseract.exe" for base in program_files if base
        )
    else:
        candidates.extend(
            [
                Path("/usr/local/bin/tesseract"),
                Path("/usr/bin/tesseract"),
                Path("/opt/homebrew/bin/tesseract"),
                Path("/opt/local/bin/tesseract"),
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None
