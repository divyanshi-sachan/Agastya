"""Tiered text extraction for contracts (PDF/image/txt)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO

import numpy as np
import pdfplumber
from PIL import Image

try:
    import easyocr
except Exception:  # pragma: no cover - optional runtime dependency in CI
    easyocr = None

try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover - optional runtime dependency in CI
    convert_from_bytes = None

_OCR_READER = None


def _get_reader():
    global _OCR_READER
    if _OCR_READER is None:
        if easyocr is None:
            raise RuntimeError("easyocr is not installed. Install phase-3 dependencies first.")
        _OCR_READER = easyocr.Reader(["en"], gpu=False)
    return _OCR_READER


def extract_text(file_input: BinaryIO) -> str:
    """
    Universal contract text extractor.

    Accepts: digital PDF, scanned PDF, PNG, JPG, TXT.
    Returns: cleaned contract text.
    """
    suffix = Path(getattr(file_input, "name", "")).suffix.lower()
    raw_bytes = file_input.read()

    if suffix == ".txt":
        return _clean_text(raw_bytes.decode("utf-8", errors="ignore"))

    if suffix == ".pdf":
        text = _extract_digital_pdf(raw_bytes)
        if len(text.strip()) > 100:
            return _clean_text(text)
        return _clean_text(_extract_pdf_ocr(raw_bytes))

    if suffix in {".png", ".jpg", ".jpeg"}:
        image = np.array(Image.open(BytesIO(raw_bytes)))
        text = " ".join(_get_reader().readtext(image, detail=0))
        return _clean_text(text)

    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def _extract_digital_pdf(raw_bytes: bytes) -> str:
    with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _extract_pdf_ocr(raw_bytes: bytes) -> str:
    if convert_from_bytes is None:
        raise RuntimeError("pdf2image is not installed. Install phase-3 dependencies first.")
    images = convert_from_bytes(raw_bytes)
    reader = _get_reader()
    chunks = [" ".join(reader.readtext(np.array(img), detail=0)) for img in images]
    return "\n".join(chunks)


def _clean_text(text: str) -> str:
    """Fix common OCR artifacts: whitespace/hyphenation/non-ascii."""
    text = re.sub(r"-\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

