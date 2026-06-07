import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io


def preprocess(image: Image.Image) -> Image.Image:
    """
    Enhance the image before OCR to improve handwriting recognition:
    1. Convert to grayscale — removes colour noise
    2. Upscale — Tesseract performs better on larger images (300+ DPI equivalent)
    3. Sharpen — makes pen strokes crisper
    4. Increase contrast — separates ink from paper more clearly
    5. Slightly increase brightness — lifts faint pencil/light ink
    """

    image = image.convert("L")

    max_side = max(image.width, image.height)
    if max_side < 2400:
        scale = 2400 / max_side
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.LANCZOS,
        )

    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Brightness(image).enhance(1.2)

    return image


def extract_text_from_bytes(image_bytes: bytes) -> str:
    """Extract handwritten text from image bytes using Tesseract with preprocessing."""
    image = Image.open(io.BytesIO(image_bytes))
    image = preprocess(image)
    text = pytesseract.image_to_string(image, config="--psm 6 --oem 1")
    return text.strip()
