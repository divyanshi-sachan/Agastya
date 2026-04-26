from __future__ import annotations

from io import BytesIO

import pytest

from src.phase3.ocr.extractor import _clean_text, extract_text


def _named_bytes(data: bytes, name: str) -> BytesIO:
    buff = BytesIO(data)
    buff.name = name
    return buff


def test_extract_text_from_txt_is_non_empty():
    file_obj = _named_bytes(b"Payment shall be made within 30 days.", "sample.txt")
    extracted = extract_text(file_obj)
    assert isinstance(extracted, str)
    assert extracted.strip()


def test_clean_text_collapses_newlines():
    cleaned = _clean_text("Line1\n\n\n\nLine2")
    assert cleaned == "Line1\n\nLine2"


def test_unsupported_type_raises():
    file_obj = _named_bytes(b"{}", "config.json")
    with pytest.raises(ValueError):
        extract_text(file_obj)

