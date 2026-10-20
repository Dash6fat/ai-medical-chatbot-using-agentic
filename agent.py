import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from config import OPENAI_MODEL
from knowledge import MEDICAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def generate_reply(
    client: AsyncOpenAI,
    message: str,
    history: list[dict[str, str]],
    images: list[str] | None = None,
) -> str:
    user_content: str | list[dict[str, Any]] = message
    if images:
        user_content = [
            {"type": "text", "text": message or "Please describe this image."},
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": image, "detail": "auto"},
                }
                for image in images
            ],
        ]

    try:
        completion = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": user_content},
            ],
            max_tokens=800,
        )
    except OpenAIError:
        logger.exception("OpenAI request failed")
        raise

    reply = completion.choices[0].message.content
    if not reply or not reply.strip():
        raise RuntimeError("The model returned an empty response.")
    return reply.strip()