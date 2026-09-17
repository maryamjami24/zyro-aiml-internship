import pytesseract
from PIL import Image, ImageOps, ImageFilter


def preprocess_image(image):
    """
    Preprocess an image to improve OCR results.
    """

    # Convert image to grayscale
    image = ImageOps.grayscale(image)

    # Increase image size for better OCR recognition
    width, height = image.size
    image = image.resize((width * 2, height * 2))

    # Apply slight noise reduction
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # Apply thresholding to make text clearer
    image = image.point(lambda pixel: 0 if pixel < 150 else 255)

    return image


def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR with preprocessing.
    """

    try:
        image = Image.open(image_path)

        # Preprocess image before OCR
        processed_image = preprocess_image(image)

        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(processed_image)

        return text.strip()

    except Exception as e:
        return f"Error processing image: {e}"


if __name__ == "__main__":
    print("OCR Processor is ready!")