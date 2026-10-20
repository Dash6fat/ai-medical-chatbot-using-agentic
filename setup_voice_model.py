import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 200 * 1024 * 1024


def main() -> None:
    project_dir = Path(__file__).parent
    models_dir = project_dir / "models"
    model_dir = models_dir / MODEL_NAME
    if model_dir.is_dir():
        print(f"Voice model is already installed: {model_dir}")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir) as temporary_dir:
        archive_path = Path(temporary_dir) / f"{MODEL_NAME}.zip"
        print(f"Downloading the offline English speech model from {MODEL_URL}")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, archive_path.open("wb") as archive:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise RuntimeError("Downloaded model archive exceeded the 100 MB safety limit.")
                archive.write(chunk)

        staging_dir = Path(temporary_dir) / "extracted"
        staging_dir.mkdir()
        extracted_bytes = 0
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError("Model archive contains an unsafe file path.")
                extracted_bytes += entry.file_size
                if extracted_bytes > MAX_EXTRACTED_BYTES:
                    raise RuntimeError("Extracted model exceeded the 200 MB safety limit.")
                destination = (staging_dir / path).resolve()
                if not destination.is_relative_to(staging_dir.resolve()):
                    raise RuntimeError("Model archive contains an unsafe file path.")
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)

        extracted_model = staging_dir / MODEL_NAME
        if not extracted_model.is_dir():
            raise RuntimeError("Downloaded archive did not contain the expected English model.")
        shutil.move(str(extracted_model), str(model_dir))
        print(f"Offline voice model installed: {model_dir}")


if __name__ == "__main__":
    main()
