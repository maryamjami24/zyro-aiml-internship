import fitz
import re
from pathlib import Path
import pytesseract
from PIL import Image, ImageOps, ImageFilter


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    return "\n".join(lines).strip()


def preprocess_image(image):
    image = ImageOps.grayscale(image)

    width, height = image.size
    image = image.resize((width * 2, height * 2))

    image = image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    image = image.point(
        lambda pixel: 0 if pixel < 150 else 255
    )

    return image


def ocr_pdf_pages(file_path):
    """
    Convert scanned PDF pages into images
    and extract text using Tesseract OCR.
    """

    text = ""

    try:
        pdf = fitz.open(file_path)

        for page in pdf:

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples
            )

            processed_image = preprocess_image(image)

            page_text = pytesseract.image_to_string(
                processed_image
            )

            text += page_text + "\n"

        pdf.close()

        return clean_text(text)

    except Exception as e:
        return f"Error processing scanned PDF: {e}"


def extract_text_from_pdf_with_method(file_path):
    """
    Extract PDF text and return both text and reading method.
    """

    text = ""

    try:
        pdf = fitz.open(file_path)

        for page in pdf:
            text += page.get_text() + "\n"

        pdf.close()

        cleaned_text = clean_text(text)

        # Normal PDF with readable text
        if len(cleaned_text.strip()) >= 30:
            return cleaned_text, "PDF text extraction"

        # Scanned/image-only PDF
        ocr_text = ocr_pdf_pages(file_path)

        return ocr_text, "OCR (scanned PDF)"

    except Exception as e:
        return f"Error processing PDF: {e}", "Error"


def extract_text_from_pdf(file_path):
    """
    Existing PDF extraction function.
    Kept compatible with the classifier.
    """

    text, method = extract_text_from_pdf_with_method(
        file_path
    )

    return text


def get_document_info(file_path):
    """
    Get basic information about the document.
    """

    path = Path(file_path)

    if not path.exists():
        return {
            "name": path.name,
            "size": 0,
            "type": "Unknown"
        }

    return {
        "name": path.name,
        "size": path.stat().st_size,
        "type": path.suffix.lower()
    }


if __name__ == "__main__":
    print("Document Processor is ready!")