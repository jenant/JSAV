"""
OCR is now handled inside analysis.py via Gemini's vision capability.
This module is kept as a no-op shim so existing imports don't break.
"""

def extract_text_from_bytes(image_bytes: bytes) -> str:
    raise NotImplementedError(
        "Standalone OCR is not used — Gemini reads images directly in analysis.py"
    )
