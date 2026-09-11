import pytesseract
from PIL import Image


def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)

        return text.strip()

    except Exception as e:
        return f"Error processing image: {e}"


if __name__ == "__main__":
    print("OCR Processor is ready!")