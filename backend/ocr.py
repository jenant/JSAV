import pytesseract
from PIL import Image
import io


def extract_text_from_bytes(image_bytes: bytes) -> str:
    """Extract handwritten text from image bytes using Tesseract (runs locally, free)."""
    image = Image.open(io.BytesIO(image_bytes))

    # PSM 6 = assume a uniform block of text — works well for journal pages
    text = pytesseract.image_to_string(image, config="--psm 6")
    return text.strip()
