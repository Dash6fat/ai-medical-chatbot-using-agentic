import base64
import binascii
import io
import math
import os
import re
import shutil
import warnings
from pathlib import Path

import pymupdf as fitz
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import TesseractError, TesseractNotFoundError

from config import (
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_PAGES,
    MAX_DOCUMENT_TEXT_CHARS,
    MAX_IMAGE_BYTES,
)

OCR_TIMEOUT_SECONDS = 8
MAX_IMAGE_PIXELS = 30_000_000
MAX_PDF_OCR_UNITS = 3
MAX_VISION_PAGES = 3
MAX_VISION_PAGE_PIXELS = 2_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

if shutil.which("tesseract") is None and os.name == "nt":
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Tesseract-OCR"
        / "tesseract.exe",
    ]
    installed_tesseract = next((path for path in candidates if path.is_file()), None)
    if installed_tesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = str(installed_tesseract)


class DocumentReadError(ValueError):
    pass


class OcrUnavailableError(RuntimeError):
    pass


def _ocr(image: Image.Image) -> str:
    prepared = ImageOps.autocontrast(image.convert("L"))
    if max(prepared.size) < 1400:
        scale = min(1.75, 1400 / max(prepared.size))
        prepared = prepared.resize(
            (round(prepared.width * scale), round(prepared.height * scale)),
            Image.Resampling.LANCZOS,
        )
    try:
        data = pytesseract.image_to_data(
            prepared,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Text recognition needs the Tesseract OCR engine. Install it and restart the app."
        ) from exc
    except TesseractError as exc:
        raise DocumentReadError("The OCR engine could not read this page.") from exc
    except RuntimeError as exc:
        raise DocumentReadError("OCR took too long to process this page.") from exc

    lines: list[str] = []
    confidences: list[float] = []
    for text, confidence in zip(data["text"], data["conf"], strict=True):
        cleaned = text.strip()
        try:
            confidence_score = float(confidence)
        except (TypeError, ValueError):
            continue
        if cleaned and confidence_score >= 45:
            lines.append(cleaned)
            confidences.append(confidence_score)
    if not lines or sum(confidences) / len(confidences) < 50:
        return ""
    return " ".join(lines)


def _read_image(content: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(content))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise DocumentReadError("Image dimensions exceed the 30-megapixel limit.")
        image.verify()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise DocumentReadError("The uploaded file is not a supported, valid image.") from exc

    text = _ocr(image)[:MAX_DOCUMENT_TEXT_CHARS]
    dimensions = f"{image.width} × {image.height} pixels"
    if not text:
        return (
            f"Image received ({dimensions}), but no sufficiently clear text was detected. "
            "Offline mode can extract visible text; it cannot identify objects, "
            "interpret scans, or diagnose a medical condition from the picture."
        )
    return (
        f"Text detected in the image ({dimensions}; OCR may contain errors):\n{text}\n\n"
        "This is text extraction only. It does not identify objects or diagnose "
        "a condition from the image."
    )


def _pdf_data_uri_content(document_data_uri: str) -> bytes:
    match = re.fullmatch(r"data:application/pdf;base64,([A-Za-z0-9+/]*={0,2})", document_data_uri)
    if match is None:
        raise DocumentReadError("Upload a valid base64-encoded PDF document.")
    content = base64.b64decode(match.group(1), validate=True)
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentReadError("PDF files must be 20 MB or smaller.")
    if not content.startswith(b"%PDF-"):
        raise DocumentReadError("The uploaded document is not a valid PDF.")
    return content


def _render_pdf_page(page: fitz.Page) -> str:
    rect = page.rect
    scale = min(1.25, math.sqrt(MAX_VISION_PAGE_PIXELS / (rect.width * rect.height)))
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _read_pdf_parts(
    content: bytes,
    *,
    include_vision_pages: bool = False,
    tolerate_missing_ocr: bool = False,
) -> tuple[str, list[str]]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, ValueError) as exc:
        raise DocumentReadError("The uploaded PDF could not be opened.") from exc

    extracted: list[str] = []
    vision_pages: list[str] = []
    ocr_unavailable = False
    ocr_skipped = False
    ocr_units = 0
    total_pages = document.page_count
    try:
        if document.needs_pass:
            raise DocumentReadError("This PDF is password-protected and cannot be read.")
        if document.page_count == 0:
            raise DocumentReadError("The uploaded PDF contains no pages.")

        page_limit = min(total_pages, MAX_DOCUMENT_PAGES)
        text_length = 0
        for index in range(page_limit):
            page = document.load_page(index)
            page_text = page.get_text("text").strip()
            image_infos = page.get_image_info(xrefs=True)
            should_render_page = bool(image_infos) or not page_text
            if include_vision_pages and should_render_page and len(vision_pages) < MAX_VISION_PAGES:
                vision_pages.append(_render_pdf_page(page))

            page_ocr: list[str] = []
            needs_ocr = bool(image_infos) or not page_text
            ocr_sources: list[bytes] = []
            if needs_ocr and ocr_units < MAX_PDF_OCR_UNITS:
                if image_infos:
                    for image_info in image_infos[:MAX_PDF_OCR_UNITS - ocr_units]:
                        image_rect = fitz.Rect(image_info["bbox"]) & page.rect
                        if image_rect.is_empty or image_rect.is_infinite:
                            continue
                        scale = min(
                            1.5,
                            math.sqrt(MAX_IMAGE_PIXELS / (image_rect.width * image_rect.height)),
                        )
                        pixmap = page.get_pixmap(
                            matrix=fitz.Matrix(scale, scale),
                            clip=image_rect,
                            alpha=False,
                        )
                        ocr_sources.append(pixmap.tobytes("png"))
                else:
                    rect = page.rect
                    scale = min(1.5, math.sqrt(MAX_IMAGE_PIXELS / (rect.width * rect.height)))
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    ocr_sources = [pixmap.tobytes("png")]
                remaining_units = MAX_PDF_OCR_UNITS - ocr_units
                if len(ocr_sources) > remaining_units:
                    ocr_skipped = True
                ocr_sources = ocr_sources[:remaining_units]
            elif needs_ocr:
                ocr_skipped = True

            for source in ocr_sources:
                try:
                    with Image.open(io.BytesIO(source)) as image:
                        if image.width * image.height > MAX_IMAGE_PIXELS:
                            continue
                        ocr_units += 1
                        page_ocr.append(_ocr(image.convert("RGB")))
                except OcrUnavailableError:
                    if not tolerate_missing_ocr:
                        raise
                    ocr_unavailable = True
                    break

            text = "\n".join(value for value in (page_text, *page_ocr) if value).strip()
            if text:
                remaining = MAX_DOCUMENT_TEXT_CHARS - text_length
                if remaining <= 0:
                    break
                page_text = text[:remaining]
                extracted.append(f"Page {index + 1}: {page_text}")
                text_length += len(page_text)
            if text_length >= MAX_DOCUMENT_TEXT_CHARS:
                break
    finally:
        document.close()

    result = "\n\n".join(extracted)[:MAX_DOCUMENT_TEXT_CHARS]
    if ocr_unavailable:
        result += (
            "\n\nOCR is unavailable on this server. Text in scanned pages and PDF figures "
            "may be missing; the selected page images are still available to the configured AI model."
        )
    if ocr_skipped:
        result += (
            f"\n\nTo keep processing quick, OCR was limited to {MAX_PDF_OCR_UNITS} "
            "scanned pages or embedded figures."
        )
    if total_pages > page_limit:
        result += f"\n\nOnly the first {page_limit} pages were read."
    if not result.strip():
        result = (
            "The PDF opened, but no readable text was found in its first "
            f"{page_limit} pages. It may contain low-quality scans. "
            "Offline mode can extract text, but cannot diagnose a condition."
        )
    else:
        result = (
            f"Text extracted from the PDF (OCR may contain errors):\n{result}\n\n"
            "This is document text extraction, not medical interpretation or diagnosis. "
            "Check the original pages for mistakes."
        )
    return result, vision_pages


def _read_pdf(content: bytes) -> str:
    return _read_pdf_parts(content)[0]


def prepare_pdf_for_vision(document_data_uri: str) -> tuple[str, list[str]]:
    content = _pdf_data_uri_content(document_data_uri)
    return _read_pdf_parts(content, include_vision_pages=True, tolerate_missing_ocr=True)


def analyze_upload(image_data_uri: str | None, document_data_uri: str | None) -> str:
    try:
        if document_data_uri is not None:
            return _read_pdf(_pdf_data_uri_content(document_data_uri))

        if image_data_uri is None:
            raise DocumentReadError("No image or document was provided.")
        match = re.fullmatch(
            r"data:image/(?:jpeg|png|webp|gif);base64,([A-Za-z0-9+/]*={0,2})",
            image_data_uri,
        )
        if match is None:
            raise DocumentReadError("Upload a valid JPEG, PNG, WebP, or GIF image.")
        content = base64.b64decode(match.group(1), validate=True)
        if not content or len(content) > MAX_IMAGE_BYTES:
            raise DocumentReadError("Image files must be non-empty and 5 MB or smaller.")
        return _read_image(content)
    except binascii.Error as exc:
        raise DocumentReadError("The uploaded file contains invalid base64 data.") from exc
