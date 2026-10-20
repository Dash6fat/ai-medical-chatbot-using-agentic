# Medical Agentic Chatbot

A local web chat for general health information. Conversations are stored in a
local SQLite database. Without an API key, the app provides a small built-in
offline guide for common topics and OCR text extraction for images and PDFs.
An OpenAI API key enables conversational AI and optional image analysis. OCR
cannot identify objects or interpret medical images; it may also misread text.
This is not a diagnostic tool and does not replace a qualified medical professional.

## Requirements

- Python 3.10 or newer
- Tesseract OCR (needed for image OCR and scanned PDFs)
- An OpenAI API key (optional; needed for AI answers and optional image analysis)

## Run on Windows

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_voice_model.py
```

Install the Tesseract OCR engine separately on Windows and ensure
`tesseract.exe` is on `PATH`. The Python `pytesseract` package is only a
wrapper; it does not install the OCR engine. Text-based PDFs can be read
without Tesseract, but image OCR and scanned PDF pages require it.

Start the server without an API key to use the built-in offline guide:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Open http://127.0.0.1:8001 in your browser. Attach a JPEG, PNG, WebP, GIF, or
PDF to extract visible text offline; PDF OCR checks up to three scanned pages
or embedded figure images and may contain errors. It cannot identify objects, interpret medical
images, or diagnose disease. With a configured vision-capable OpenAI model,
the app can summarize up to three figure-bearing PDF pages. This is optional
and requires an API key; it does not train a model or provide a diagnosis.
Create a `.env` file in this folder and add your key before starting the server.
Low-confidence OCR words are filtered to reduce garbled labels; captions that
remain are still unverified and must be checked against the original image.

To use a `.env` file instead, create it in this folder with:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
```

Chat history is saved to `data/chat_history.sqlite3`. Set `CHATBOT_DB_PATH` to
choose a different SQLite database file. The database is created automatically
and is excluded from Git.

Camera access works on `localhost` or over HTTPS and requires permission.
The one-time `setup_voice_model.py` command downloads the small English
offline speech model (about 40 MB). Voice recordings up to 10 minutes are
processed by the app's server; recordings are not sent to an external speech
service or saved in the database.
Set `VOSK_MODEL_PATH` to use a different compatible Vosk model. Read-aloud uses
the browser/device's built-in speech synthesis.

Do not commit `.env` or share your API key. Pinecone is not required.
Medical references and the textbook's citation-only treatment are described
in [REFERENCES.md](./REFERENCES.md).

## Prepare a public Render deployment

The project includes a Docker image and a Render Blueprint in `render.yaml`.
For the public demo, the Blueprint disables server-side conversation
persistence by default; Render's free instance also has ephemeral storage.
This is intentional for visitor privacy. Do not enable public health-chat
retention without user authentication, a privacy/retention policy, and
appropriate legal and security review.

1. Create a GitHub repository containing only this project folder and push it.
   Do not push the `C:\Users\dell` home-directory repository. Do not include the
   textbook PDF, `.env`, API keys, local database, or downloaded voice model.
2. In the Render Dashboard, create a Blueprint from that repository and select
   the project's `render.yaml` file.
3. After deployment, test the generated `https://<service-name>.onrender.com`
   URL and its `/health` endpoint.
4. In Render's service **Settings → Custom Domains**, add the exact hostname
   `venkannaoletimedicalchatbot.ai.com`. This is a subdomain of `ai.com`, so
   DNS changes must be made by whoever controls that parent domain's DNS zone.
   Follow Render's displayed DNS instructions and wait for verification and
   automatic TLS certificate provisioning. Render redirects HTTP traffic to
   HTTPS; do not publish health conversations over plain HTTP.
5. An OpenAI key is optional for the public demo. To enable AI replies, configure
   `OPENAI_API_KEY` as a secret in Render's service environment; never commit it
   or paste it into chat.

This repository currently has no Git remote or hosting credentials configured,
and the requested subdomain does not resolve in public DNS yet. Creating the
deployment files does not itself publish a website or prove DNS control.

## Endpoints

- `GET /` — chat interface
- `GET /health` — application, database, and AI configuration status
- `POST /api/chat` — send a message and an optional saved conversation ID
- `GET /api/conversations/{id}` — restore a saved conversation