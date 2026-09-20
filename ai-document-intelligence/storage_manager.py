import hashlib
import re
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

INVOICE_DIR = STORAGE_DIR / "invoices"
RESUME_DIR = STORAGE_DIR / "resumes"
OTHER_DIR = STORAGE_DIR / "other"


def initialize_storage():
    """Create organized storage folders."""
    INVOICE_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    OTHER_DIR.mkdir(parents=True, exist_ok=True)


def get_storage_directory(document_type):
    """Return the correct storage folder for a document type."""
    document_type = (document_type or "").strip().lower()

    if document_type == "invoice":
        return INVOICE_DIR

    if document_type == "resume":
        return RESUME_DIR

    return OTHER_DIR


def create_safe_filename(original_filename):
    """Create a safe unique filename while preserving the original extension."""
    original_filename = Path(original_filename).name

    extension = Path(original_filename).suffix.lower()

    base_name = Path(original_filename).stem

    # Replace unsafe characters with underscores.
    safe_base = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name)

    # Prevent excessively long filenames.
    safe_base = safe_base[:60].strip("_")

    if not safe_base:
        safe_base = "document"

    unique_id = uuid.uuid4().hex[:8]

    return f"{safe_base}_{unique_id}{extension}"


def calculate_file_hash(file_bytes):
    """Calculate SHA-256 hash for uploaded file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def save_file(file_bytes, original_filename, document_type):
    """
    Save a file into the correct storage folder.

    Returns:
        tuple: (stored_filename, file_path)
    """
    initialize_storage()

    storage_directory = get_storage_directory(document_type)
    stored_filename = create_safe_filename(original_filename)

    file_path = storage_directory / stored_filename

    file_path.write_bytes(file_bytes)

    return stored_filename, file_path