import fitz
from pathlib import Path


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF document.
    """
    text = ""

    try:
        pdf = fitz.open(file_path)

        for page in pdf:
            text += page.get_text()

        pdf.close()

        return text.strip()

    except Exception as e:
        return f"Error processing PDF: {e}"


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