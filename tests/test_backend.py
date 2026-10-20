import io
import sqlite3
import wave
import base64

import pymupdf as fitz
from PIL import Image
from fastapi.testclient import TestClient

import config
import main
from document_reader import OcrUnavailableError, _ocr, analyze_upload, prepare_pdf_for_vision
from knowledge import offline_reply
from voice import transcribe_wav


def test_chat_works_offline_and_persists_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "database": "ok",
            "openai_configured": False,
            "chat_mode": "offline",
            "voice_configured": main.app.state.voice_model is not None,
            "image_text_extraction": "tesseract-ocr",
            "conversation_persistence_enabled": True,
        }

        response = client.post("/api/chat", json={"message": "I have a headache"})
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == "offline"
        assert "Headaches can have many causes" in result["reply"]

        saved = client.get(f"/api/conversations/{result['conversation_id']}")
        assert saved.status_code == 200
        assert [message["role"] for message in saved.json()["messages"]] == [
            "user",
            "assistant",
        ]
        assert saved.json()["messages"][0]["content"] == "I have a headache"


def test_chat_offline_mode_rejects_invalid_image_and_conversation_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")

    with TestClient(main.app) as client:
        invalid_image = client.post(
            "/api/chat",
            json={"message": "What is this?", "image": "data:image/png;base64,!!!"},
        )
        assert invalid_image.status_code == 422
        invalid_conversation = client.post(
            "/api/chat",
            json={"message": "hello", "conversation_id": "not-a-uuid"},
        )
        assert invalid_conversation.status_code == 422


def test_offline_chat_extracts_image_text_and_adds_general_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "analyze_upload", lambda image, document: "Text detected: clinic hours")
    image = "data:image/png;base64," + base64.b64encode(b"image bytes").decode()

    with TestClient(main.app) as client:
        response = client.post("/api/chat", json={"message": "I have a headache", "image": image})

    assert response.status_code == 200
    assert "Headaches can have many causes" in response.json()["reply"]
    assert "clinic hours" in response.json()["reply"]
    assert "Visual summaries are unavailable in offline mode" in response.json()["reply"]
    assert "OPENAI_API_KEY" in response.json()["reply"]


def test_offline_image_question_does_not_get_unrelated_general_symptom_script(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "analyze_upload", lambda image, document: "OCR text: possible X-ray caption")
    image = "data:image/png;base64," + base64.b64encode(b"image bytes").decode()

    with TestClient(main.app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "What does this X-ray show?", "image": image},
        )

    reply = response.json()["reply"]
    assert response.status_code == 200
    assert "cannot visually interpret an uploaded image" in reply
    assert "unverified reading" in reply
    assert "possible X-ray caption" in reply
    assert "Try asking about a headache" not in reply


def test_ocr_filters_words_below_confidence_threshold(monkeypatch):
    monkeypatch.setattr(
        "document_reader.pytesseract.image_to_data",
        lambda *args, **kwargs: {
            "text": ["clear", "noise", "label"],
            "conf": ["90", "20", "80"],
        },
    )

    assert _ocr(Image.new("RGB", (100, 50), "white")) == "clear label"


def test_offline_chat_reads_uploaded_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "analyze_upload", lambda image, document: "Text extracted: appointment details")
    pdf = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.7 test").decode()

    with TestClient(main.app) as client:
        response = client.post("/api/chat", json={"message": "Summarize this", "document": pdf})

    assert response.status_code == 200
    assert "appointment details" in response.json()["reply"]
    assert "diagnose" in response.json()["reply"].lower()


def test_pdf_validation_rejects_invalid_signature_and_oversized_files(monkeypatch):
    invalid_pdf = "data:application/pdf;base64," + base64.b64encode(b"not a PDF").decode()
    with TestClient(main.app) as client:
        invalid = client.post("/api/chat", json={"message": "Read this", "document": invalid_pdf})
        monkeypatch.setattr(main, "MAX_DOCUMENT_BYTES", 3)
        oversized = "data:application/pdf;base64," + base64.b64encode(b"%PDF-data").decode()
        too_large = client.post("/api/chat", json={"message": "Read this", "document": oversized})

    assert invalid.status_code == 422
    assert too_large.status_code == 422


def test_offline_upload_explains_missing_tesseract(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")

    def missing_ocr_engine(image, document):
        raise OcrUnavailableError("Install Tesseract OCR.")

    monkeypatch.setattr(main, "analyze_upload", missing_ocr_engine)
    image = "data:image/png;base64," + base64.b64encode(b"image bytes").decode()

    with TestClient(main.app) as client:
        response = client.post("/api/chat", json={"message": "Read this", "image": image})

    assert response.status_code == 503
    assert response.json()["detail"] == "Install Tesseract OCR."


def test_text_pdf_can_be_read_without_running_ocr():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Follow up with your clinician")
    content = pdf.tobytes()
    pdf.close()
    data_uri = "data:application/pdf;base64," + base64.b64encode(content).decode()

    result = analyze_upload(None, data_uri)

    assert "Follow up with your clinician" in result
    assert "not medical interpretation or diagnosis" in result


def test_pdf_ocr_reads_figure_images_on_pages_with_text(monkeypatch):
    figure = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(figure, format="PNG")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Figure 1")
    page.insert_image(fitz.Rect(72, 90, 152, 130), stream=figure.getvalue())
    content = pdf.tobytes()
    pdf.close()
    data_uri = "data:application/pdf;base64," + base64.b64encode(content).decode()
    monkeypatch.setattr("document_reader._ocr", lambda image: "recognized figure label")

    result = analyze_upload(None, data_uri)

    assert "Figure 1" in result
    assert "recognized figure label" in result


def test_pdf_vision_prepares_embedded_figure_page_without_tesseract(monkeypatch):
    figure = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(figure, format="PNG")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Figure 1")
    page.insert_image(fitz.Rect(72, 90, 152, 130), stream=figure.getvalue())
    content = pdf.tobytes()
    pdf.close()
    data_uri = "data:application/pdf;base64," + base64.b64encode(content).decode()

    def unavailable_ocr(image):
        raise OcrUnavailableError("Tesseract is missing")

    monkeypatch.setattr("document_reader._ocr", unavailable_ocr)
    extracted, pages = prepare_pdf_for_vision(data_uri)

    assert "OCR is unavailable" in extracted
    assert len(pages) == 1
    assert pages[0].startswith("data:image/jpeg;base64,")


def test_ai_chat_sends_pdf_page_images_to_vision_model(monkeypatch):
    monkeypatch.setattr(main, "PERSIST_CHAT_HISTORY", False)
    pdf_image = "data:image/jpeg;base64,ZmFrZQ=="
    monkeypatch.setattr(main, "prepare_pdf_for_vision", lambda document: ("PDF context", [pdf_image]))
    captured = {}

    class Client:
        async def close(self):
            pass

    async def fake_generate_reply(client, message, history, images):
        captured["message"] = message
        captured["images"] = images
        return "Visual summary"

    monkeypatch.setattr(main, "generate_reply", fake_generate_reply)
    pdf_data = "data:application/pdf;base64," + base64.b64encode(b"%PDF-test").decode()
    with TestClient(main.app) as client:
        client.app.state.openai_client = Client()
        response = client.post("/api/chat", json={"message": "Explain the figure", "document": pdf_data})

    assert response.status_code == 200
    assert response.json()["reply"] == "Visual summary"
    assert "PDF context" in captured["message"]
    assert captured["images"] == [pdf_image]


def test_public_mode_does_not_persist_conversations(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "PERSIST_CHAT_HISTORY", False)

    with TestClient(main.app) as client:
        response = client.post("/api/chat", json={"message": "I have a cough"})
        assert response.status_code == 200
        result = response.json()
        assert result["conversation_id"] is None
        assert result["mode"] == "offline"
        assert client.get("/health").json()["conversation_persistence_enabled"] is False
        assert client.get("/api/conversations/00000000-0000-0000-0000-000000000001").status_code == 404

        with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
            count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 0


def test_medical_reference_is_citation_only_and_publicly_accessible():
    with TestClient(main.app) as client:
        response = client.get("/REFERENCES.md")
        assert response.status_code == 200
        assert "Current Medical Diagnosis & Treatment 2025" in response.text
        assert "has not been" in response.text


def test_offline_guidance_prioritizes_emergencies():
    reply = offline_reply("I have chest pain and cannot breathe")
    assert "emergency" in reply.lower()
    assert "local emergency services" in reply.lower()


def test_voice_endpoint_transcribes_local_wav(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_DB_PATH", str(tmp_path / "chat.sqlite3"))
    monkeypatch.setattr(main, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "transcribe_wav", lambda model, audio: "describe my headache")

    with TestClient(main.app) as client:
        client.app.state.voice_model = object()
        response = client.post(
            "/api/voice",
            content=b"recording",
            headers={"Content-Type": "audio/wav"},
        )
        assert response.status_code == 200
        assert response.json() == {"transcript": "describe my headache"}

        unsupported_type = client.post(
            "/api/voice",
            content=b"recording",
            headers={"Content-Type": "audio/webm"},
        )
        assert unsupported_type.status_code == 415


def test_voice_recognizer_validates_recording_format():
    recording = io.BytesIO()
    with wave.open(recording, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0" * 8)

    try:
        transcribe_wav(object(), recording.getvalue())
    except ValueError as exc:
        assert "mono 16 kHz" in str(exc)
    else:
        raise AssertionError("Stereo audio should be rejected.")


def test_voice_limit_is_ten_minutes_and_accepts_maximum_duration():
    assert config.MAX_VOICE_SECONDS == 600
    recording = io.BytesIO()
    with wave.open(recording, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0" * (config.MAX_VOICE_SECONDS * 16000 * 2))

    with wave.open(io.BytesIO(recording.getvalue()), "rb") as audio:
        assert audio.getnframes() == config.MAX_VOICE_SECONDS * 16000
