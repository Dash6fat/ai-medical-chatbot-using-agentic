FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VOSK_MODEL_PATH=/opt/chatbot/models/vosk-model-small-en-us-0.15 \
    CHATBOT_DB_PATH=/tmp/chat_history.sqlite3 \
    CHATBOT_PERSIST_HISTORY=false

WORKDIR /opt/chatbot

COPY requirements.txt setup_voice_model.py ./
RUN apt-get update \
    && apt-get install --no-install-recommends -y tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .
RUN python setup_voice_model.py

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
