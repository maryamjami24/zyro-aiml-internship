import re
from pathlib import Path

import joblib
import streamlit as st

from database import (
    initialize_database,
    add_document,
    get_document_by_hash,
    get_document,
    filter_documents,
)

from storage_manager import (
    initialize_storage,
    calculate_file_hash,
    save_file,
)

from document_processor import (
    extract_text_from_pdf,
    extract_text_from_pdf_with_method,
    clean_text
)

from ocr_processor import (
    extract_text_from_image
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

MODEL_FILE = Path("document_classifier.pkl")

classifier_model = None

if MODEL_FILE.exists():
    try:
        classifier_model = joblib.load(MODEL_FILE)
    except Exception:
        classifier_model = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_spaces(text):
    """
    Normalize unnecessary spaces and line breaks.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n\s*\n+", "\n\n", text)

    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    return "\n".join(lines).strip()


# ============================================================
# RULE-BASED DOCUMENT CLASSIFIER
# ============================================================

def classify_document_rule_based(text):
    """
    Simple keyword-based baseline classifier.
    """

    text_lower = text.lower()

    invoice_keywords = [
        "invoice",
        "invoice number",
        "invoice no",
        "invoice date",
        "amount due",
        "balance due",
        "subtotal",
        "total amount",
        "bill to",
        "unit price",
        "quantity",
        "tax",
        "proforma invoice",
        "grand total"
    ]

    resume_keywords = [
        "resume",
        "curriculum vitae",
        "profile",
        "employment history",
        "work experience",
        "professional experience",
        "education",
        "skills",
        "certifications",
        "achievements"
    ]

    invoice_score = sum(
        1
        for keyword in invoice_keywords
        if keyword in text_lower
    )

    resume_score = sum(
        1
        for keyword in resume_keywords
        if keyword in text_lower
    )

    if invoice_score >= 2 and invoice_score > resume_score:
        return "Invoice"

    if resume_score >= 2 and resume_score > invoice_score:
        return "Resume"

    return "Other"


# ============================================================
# MACHINE LEARNING CLASSIFIER
# ============================================================

def classify_document_ml(text):
    """
    Classify the document using the trained ML model.
    """

    if classifier_model is None:
        return "Other", 0.0

    if not text.strip():
        return "Other", 0.0

    try:
        prediction = classifier_model.predict([text])[0]

        document_type = str(prediction).title()

        confidence = 0.0

        if hasattr(
            classifier_model,
            "predict_proba"
        ):
            probabilities = classifier_model.predict_proba(
                [text]
            )[0]

            confidence = max(probabilities) * 100

        return document_type, confidence

    except Exception:
        return "Other", 0.0


# ============================================================
# GENERIC SECTION FINDER
# ============================================================

def find_section(lines, heading_patterns, stop_patterns):
    """
    Find a section such as Skills and return its lines.
    """

    start_index = None

    for index, line in enumerate(lines):

        line_clean = line.strip().lower()

        for pattern in heading_patterns:

            if re.fullmatch(
                pattern,
                line_clean,
                re.IGNORECASE
            ):
                start_index = index
                break

        if start_index is not None:
            break

    if start_index is None:
        return []

    section_lines = []

    for line in lines[start_index + 1:]:

        line_clean = line.strip().lower()

        should_stop = False

        for pattern in stop_patterns:

            if re.fullmatch(
                pattern,
                line_clean,
                re.IGNORECASE
            ):
                should_stop = True
                break

        if should_stop:
            break

        if line.strip():
            section_lines.append(line.strip())

    return section_lines


# ============================================================
# RESUME NAME EXTRACTION
# ============================================================

def extract_resume_name(lines):
    """
    Try to identify the person's name from the beginning
    of a resume.
    """

    ignored_lines = {
        "resume",
        "curriculum vitae",
        "cv",
        "profile",
        "details",
        "personal details",
        "contact",
        "contact details",
        "skills",
        "experience",
        "employment history",
        "work experience",
        "education",
        "achievements",
        "languages",
        "hobbies"
    }

    ignored_phrases = [
        "resume template",
        "build this resume",
        "make this resume",
        "download",
        "linkedin",
        "pinterest"
    ]

    for line in lines[:15]:

        value = line.strip()
        value_lower = value.lower()

        if not value:
            continue

        if value_lower in ignored_lines:
            continue

        if any(
            phrase in value_lower
            for phrase in ignored_phrases
        ):
            continue

        if "@" in value:
            continue

        if re.search(r"\d", value):
            continue

        words = value.split()

        if len(words) < 1 or len(words) > 5:
            continue

        if value.isupper() and len(words) > 3:
            continue

        return value

    return "Not Found"


# ============================================================
# RESUME EXTRACTION
# ============================================================

def extract_resume_fields(text):
    """
    Generic resume information extraction.
    """

    fields = {
        "Name": "Not Found",
        "Email": "Not Found",
        "Phone": "Not Found",
        "Skills": "Not Found"
    }

    if not text or not text.strip():
        return fields

    normalized_text = normalize_spaces(text)

    lines = [
        line.strip()
        for line in normalized_text.splitlines()
        if line.strip()
    ]

    fields["Name"] = extract_resume_name(lines)

    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        normalized_text
    )

    if email_match:
        fields["Email"] = email_match.group(0)

    phone_patterns = [
        r"\+\d{1,3}"
        r"[\s\-()]?"
        r"\d{2,4}"
        r"[\s\-()]?"
        r"\d{3,4}"
        r"[\s\-()]?"
        r"\d{3,4}",

        r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",

        r"\b\d{3}\s\d{3}\s\d{4}\b",

        r"\b\d{10,15}\b"
    ]

    for pattern in phone_patterns:

        phone_match = re.search(
            pattern,
            normalized_text
        )

        if phone_match:
            fields["Phone"] = phone_match.group(0)
            break

    skill_headings = [
        r"skills",
        r"key skills",
        r"core skills",
        r"technical skills",
        r"professional skills",
        r"competencies",
        r"core competencies"
    ]

    stop_headings = [
        r"profile",
        r"summary",
        r"professional summary",
        r"objective",
        r"employment history",
        r"work experience",
        r"professional experience",
        r"experience",
        r"education",
        r"certifications",
        r"certification",
        r"achievements",
        r"hobbies",
        r"languages",
        r"references",
        r"courses",
        r"details",
        r"links"
    ]

    skill_lines = find_section(
        lines,
        skill_headings,
        stop_headings
    )

    cleaned_skills = []

    for skill in skill_lines:

        skill = skill.strip()

        if not skill:
            continue

        skill = re.sub(
            r"^[•●▪◦\-–—]+\s*",
            "",
            skill
        ).strip()

        if not skill:
            continue

        ignored_skill_text = [
            "resume templates",
            "build this resume",
            "make this resume",
            "linkedin",
            "pinterest"
        ]

        if skill.lower() in ignored_skill_text:
            continue

        if len(skill.split()) > 15:
            continue

        if skill.lower() not in [
            item.lower()
            for item in cleaned_skills
        ]:
            cleaned_skills.append(skill)

    if cleaned_skills:
        fields["Skills"] = ", ".join(
            cleaned_skills
        )

    return fields


# ============================================================
# INVOICE NUMBER EXTRACTION
# ============================================================

def extract_invoice_number(text):
    """
    Extract invoice number from common invoice formats.
    """

    if not text or not text.strip():
        return "Not Found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    same_line_patterns = [
        r"^invoice\s*(?:number|no\.?|#)"
        r"\s*[:\-]\s*"
        r"([A-Za-z0-9][A-Za-z0-9./_-]{2,})$",

        r"^invoice\s*(?:number|no\.?|#)"
        r"\s+"
        r"([A-Za-z0-9][A-Za-z0-9./_-]{2,})$",

        r"^proforma\s+invoice\s*#"
        r"\s*[:\-]\s*"
        r"([A-Za-z0-9][A-Za-z0-9./_-]{2,})$"
    ]

    for line in lines:

        for pattern in same_line_patterns:

            match = re.search(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:

                value = match.group(1).strip()

                if value.lower() not in [
                    "invoice",
                    "number",
                    "no",
                    "date"
                ]:
                    return value

    standalone_patterns = [
        r"(?<![\d-])\d{4,10}-\d{2,10}/\d{1,4}(?![\d/])",
        r"(?<![\d/])\d{4,10}/\d{1,4}(?![\d/])",
        r"\bINV[\s_-]?\d[\w./_-]*\b"
    ]

    for pattern in standalone_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0).strip()

    return "Not Found"


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_invoice_date(text):
    """
    Extract invoice/proforma invoice date.
    """

    if not text or not text.strip():
        return "Not Found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    date_patterns = [
        r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
        r"(?<!\d)\d{4}[./-]\d{1,2}[./-]\d{1,2}(?!\d)",
        r"\b[A-Za-z]+\s+\d{1,2},\s+\d{4}\b"
    ]

    labeled_patterns = [
        r"^(?:invoice\s*date|date\s*of\s*issue)"
        r"\s*[:\-]\s*"
        r"("
        r"\d{1,2}[./-]\d{1,2}[./-]\d{4}"
        r")$",

        r"^(?:invoice\s*date|date\s*of\s*issue)"
        r"\s*[:\-]\s*"
        r"("
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"
        r")$"
    ]

    for line in lines:

        for pattern in labeled_patterns:

            match = re.search(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:
                return match.group(1).strip()

    date_labels = [
        "invoice date",
        "date of issue",
        "date:"
    ]

    for index, line in enumerate(lines):

        clean_line = line.lower().strip()

        if clean_line in date_labels:

            for candidate in lines[index + 1:index + 12]:

                for pattern in date_patterns:

                    match = re.search(
                        pattern,
                        candidate
                    )

                    if match:
                        return match.group(0)

    all_dates = []

    for line in lines:

        for pattern in date_patterns:

            matches = re.findall(
                pattern,
                line
            )

            all_dates.extend(matches)

    if all_dates:
        return all_dates[0]

    return "Not Found"


# ============================================================
# COMPANY NAME EXTRACTION
# ============================================================

def extract_company_name(text):
    """
    Extract company/vendor name from common invoice formats.
    """

    if not text or not text.strip():
        return "Not Found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    explicit_patterns = [
        r"^company\s*name\s*[:\-]\s*(.+)$",
        r"^company\s*[:\-]\s*(.+)$",
        r"^vendor\s*[:\-]\s*(.+)$",
        r"^supplier\s*[:\-]\s*(.+)$",
        r"^billed\s*by\s*[:\-]\s*(.+)$"
    ]

    for line in lines:

        for pattern in explicit_patterns:

            match = re.search(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:

                value = match.group(1).strip()

                if value:
                    return value

    ignored_lines = {
        "invoice",
        "proforma invoice",
        "to:",
        "to",
        "bill to:",
        "bill to",
        "additional information",
        "description",
        "payment terms",
        "payment details",
        "company info:",
        "company info",
        "contact information",
        "registered address",
        "customer ltd.",
        "john doe"
    }

    for index, line in enumerate(lines):

        value = line.strip()

        if value.lower() in ignored_lines:
            continue

        if "@" in value:
            continue

        if re.search(r"\d", value):
            continue

        if len(value.split()) < 1:
            continue

        if len(value.split()) > 6:
            continue

        if value.lower().endswith(":"):
            continue

        if index + 1 >= len(lines):
            continue

        next_line = lines[index + 1]

        address_indicators = [
            "road",
            "street",
            "avenue",
            "lane",
            "drive",
            "boulevard",
            "address",
            "new hampshire",
            "birmingham",
            "new york"
        ]

        if (
            re.search(r"\d", next_line)
            and any(
                indicator in next_line.lower()
                for indicator in address_indicators
            )
        ):
            return value

    return "Not Found"


# ============================================================
# TOTAL AMOUNT EXTRACTION
# ============================================================

def extract_total_amount(text):
    """
    Extract the final total amount from an invoice.
    """

    if not text or not text.strip():
        return "Not Found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    amount_pattern = (
        r"(?<![\d.])"
        r"(?:[$€£]\s*)?"
        r"\d+(?:,\d{3})*"
        r"\.\d{2}"
        r"\s*(?:[$€£])?"
        r"(?![\d.])"
    )

    priority_labels = [
        "grand total",
        "total amount",
        "total due",
        "balance due",
        "amount due"
    ]

    for index, line in enumerate(lines):

        clean_line = line.lower().strip()

        for label in priority_labels:

            pattern = (
                rf"^{re.escape(label)}"
                rf"\s*[:\-]?\s*({amount_pattern})$"
            )

            match = re.search(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:
                return match.group(1).strip()

        if clean_line in priority_labels:

            for candidate in lines[index + 1:index + 6]:

                match = re.search(
                    amount_pattern,
                    candidate
                )

                if match:
                    return match.group(0).strip()

    for index, line in enumerate(lines):

        clean_line = line.lower().strip()

        if clean_line == "total":

            for candidate in lines[index + 1:index + 6]:

                match = re.search(
                    amount_pattern,
                    candidate
                )

                if match:
                    return match.group(0).strip()

    for line in lines:

        match = re.search(
            rf"^total\s*[:\-]?\s*({amount_pattern})$",
            line,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    amounts = []

    for line in lines:

        for match in re.finditer(
            amount_pattern,
            line
        ):

            value = match.group(0).strip()

            if re.fullmatch(
                r"\d{1,2}[./-]\d{1,2}[./-]\d{4}",
                value
            ):
                continue

            amounts.append(value)

    if amounts:
        return amounts[-1]

    return "Not Found"


# ============================================================
# INVOICE EXTRACTION
# ============================================================

def extract_invoice_fields(text):
    """
    Generic invoice information extraction.
    """

    fields = {
        "Invoice Number": "Not Found",
        "Date": "Not Found",
        "Company Name": "Not Found",
        "Total Amount": "Not Found"
    }

    if not text or not text.strip():
        return fields

    normalized_text = normalize_spaces(text)

    fields["Invoice Number"] = extract_invoice_number(
        normalized_text
    )

    fields["Date"] = extract_invoice_date(
        normalized_text
    )

    fields["Company Name"] = extract_company_name(
        normalized_text
    )

    fields["Total Amount"] = extract_total_amount(
        normalized_text
    )

    return fields


# ============================================================
# MAIN DOCUMENT PROCESSING
# ============================================================

def process_document(uploaded_file):

    file_name = uploaded_file.name

    extension = Path(
        file_name
    ).suffix.lower()

    temp_path = Path(
        "temp_uploaded_document" + extension
    )

    try:

        with open(
            temp_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if extension == ".pdf":

            raw_text, reading_method = (
                extract_text_from_pdf_with_method(
                    str(temp_path)
                )
            )

            if (
                not raw_text
                or raw_text.startswith(
                    "Error processing PDF"
                )
                or raw_text.startswith(
                    "Error processing scanned PDF"
                )
                or len(raw_text.strip()) < 30
            ):

                raw_text = ""

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        elif extension in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            raw_text = extract_text_from_image(
                str(temp_path)
            )

            reading_method = (
                "OCR with image preprocessing"
            )

        else:

            return {
                "success": False,
                "error": (
                    "Unsupported file type. "
                    "Please upload PDF, JPG, JPEG, or PNG."
                )
            }

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        cleaned_text = clean_text(
            raw_text
        )

        # ----------------------------------------------------
        # Empty text check
        # ----------------------------------------------------

        if not cleaned_text:

            return {
                "success": False,
                "error": (
                    "No readable text was found "
                    "in this document."
                )
            }

        if len(cleaned_text.strip()) < 20:

            return {
                "success": False,
                "error": (
                    "The extracted text is too short "
                    "to identify this document reliably."
                )
            }

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        ml_type, ml_confidence = classify_document_ml(
            cleaned_text
        )

        rule_type = classify_document_rule_based(
            cleaned_text
        )

        # ----------------------------------------------------
        # Extraction
        # ----------------------------------------------------

        invoice_fields = None
        resume_fields = None

        if ml_type == "Invoice":

            invoice_fields = extract_invoice_fields(
                cleaned_text
            )

        elif ml_type == "Resume":

            resume_fields = extract_resume_fields(
                cleaned_text
            )

        elif rule_type == "Invoice":

            invoice_fields = extract_invoice_fields(
                cleaned_text
            )

        elif rule_type == "Resume":

            resume_fields = extract_resume_fields(
                cleaned_text
            )

        # ----------------------------------------------------
        # Final document type
        # ----------------------------------------------------

        # ------------------------------------------------
        # Final document type
        # ------------------------------------------------
        # Use the ML prediction when it agrees with the
        # rule-based classifier.
        #
        # If ML and rule-based classification disagree,
        # prefer the rule-based result. This prevents
        # unrelated documents from being incorrectly
        # stored as invoices or resumes.

        if (
            ml_type in [
                "Invoice",
                "Resume"
            ]
            and ml_type == rule_type
        ):

            document_type = ml_type

        elif rule_type in [
            "Invoice",
            "Resume"
        ]:

            document_type = rule_type

        else:

            document_type = "Other"

        # ----------------------------------------------------
        # Processing status
        # ----------------------------------------------------

        status = "Processed"
        def is_missing_field(value):
            if value is None:
                return True

            value = str(value).strip().lower()

            return value in [
                "",
                "not found",
                "not_found",
                "n/a",
                "none",
                "null",
                "missing"
            ]

        if document_type == "Invoice":

            if invoice_fields is None:
                status = "Needs Review"

            else:

                missing_invoice_fields = []

                for field_name in [
                    "Invoice Number",
                    "Date",
                    "Company Name",
                    "Total Amount"
                ]:

                    value = invoice_fields.get(
                        field_name
                    )

                    if is_missing_field(value):
                        missing_invoice_fields.append(
                            field_name
                        )

                if missing_invoice_fields:
                    status = "Needs Review"

        elif document_type == "Resume":

            if resume_fields is None:
                status = "Needs Review"

            else:

                missing_resume_fields = []

                for field_name in [
                    "Name",
                    "Email",
                    "Phone"
                ]:

                    value = resume_fields.get(
                        field_name
                    )

                    if is_missing_field(value):
                        missing_resume_fields.append(
                            field_name
                        )

                if missing_resume_fields:
                    status = "Needs Review"

        return {
            "success": True,
            "file_name": file_name,
            "reading_method": reading_method,
            "cleaned_text": cleaned_text,
            "ml_type": ml_type,
            "ml_confidence": ml_confidence,
            "rule_type": rule_type,
            "document_type": document_type,
            "invoice_fields": invoice_fields,
            "resume_fields": resume_fields,
            "status": status
        }

    except Exception:

        return {
            "success": False,
            "error": (
                "The document could not be processed. "
                "Please check the file and try again."
            )
        }

    finally:

        try:

            if temp_path.exists():
                temp_path.unlink()

        except Exception:
            pass


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

initialize_database()
initialize_storage()

MAX_FILE_SIZE = 10 * 1024 * 1024

SUPPORTED_EXTENSIONS = [
    "pdf",
    "jpg",
    "jpeg",
    "png",
]


st.title("AI Document Intelligence")

st.write(
    "Upload, organize, search, and manage invoices, resumes, "
    "and other documents."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Document Management")

page = st.sidebar.radio(
    "Select Section",
    [
        "Upload Document",
        "Document Repository",
    ],
)


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

if page == "Upload Document":

    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=SUPPORTED_EXTENSIONS,
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        # ----------------------------------------------------
        # File size validation
        # ----------------------------------------------------

        if len(file_bytes) > MAX_FILE_SIZE:

            st.error(
                "File is too large. Maximum allowed size is 10 MB."
            )

        else:

            file_hash = calculate_file_hash(
                file_bytes
            )

            # ------------------------------------------------
            # Duplicate detection
            # ------------------------------------------------

            existing_document = get_document_by_hash(
                file_hash
            )

            if existing_document is not None:

                st.warning(
                    "Duplicate document detected. "
                    "This file already exists in the repository."
                )

                st.subheader("Existing Document")

                st.write(
                    "**Original filename:**",
                    existing_document["original_filename"]
                )

                st.write(
                    "**Document type:**",
                    existing_document["document_type"]
                )

                st.write(
                    "**Upload date:**",
                    existing_document["upload_date"]
                )

                st.write(
                    "**Status:**",
                    existing_document["status"]
                )

                st.write(
                    "**Stored file:**",
                    existing_document["file_path"]
                )

                st.info(
                    "No new database record or file was created."
                )

            else:

                result = process_document(
                    uploaded_file
                )

                if not result["success"]:

                    st.error(
                        result["error"]
                    )

                else:

                    # ----------------------------------------
                    # Save file after successful processing
                    # ----------------------------------------

                    stored_filename, file_path = save_file(
                        file_bytes,
                        result["file_name"],
                        result["document_type"],
                    )

                    relative_file_path = file_path.relative_to(
                        Path(__file__).resolve().parent
                    )

                    # ----------------------------------------
                    # Extract metadata
                    # ----------------------------------------

                    company = None
                    invoice_number = None
                    total_amount = None

                    if result["invoice_fields"] is not None:

                        company = result[
                            "invoice_fields"
                        ].get(
                            "Company Name"
                        )

                        invoice_number = result[
                            "invoice_fields"
                        ].get(
                            "Invoice Number"
                        )

                        total_amount = result[
                            "invoice_fields"
                        ].get(
                            "Total Amount"
                        )

                    # ----------------------------------------
                    # Add repository record
                    # ----------------------------------------

                    try:

                        document_id = add_document(
                            original_filename=result[
                                "file_name"
                            ],
                            stored_filename=stored_filename,
                            document_type=result[
                                "document_type"
                            ],
                            company=company,
                            invoice_number=invoice_number,
                            total_amount=total_amount,
                            file_path=str(
                                relative_file_path
                            ),
                            text_preview=result[
                                "cleaned_text"
                            ][:500],
                            file_hash=file_hash,
                            status=result[
                                "status"
                            ],
                        )

                        st.success(
                            "Document processed and saved successfully."
                        )

                    except Exception:

                        try:

                            if file_path.exists():
                                file_path.unlink()

                        except Exception:
                            pass

                        st.error(
                            "The document could not be saved "
                            "to the repository."
                        )

                        document_id = None

                    # ----------------------------------------
                    # Document analysis
                    # ----------------------------------------

                    st.header("Document Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "ML Document Type",
                            result["ml_type"]
                        )

                    with col2:

                        confidence = result[
                            "ml_confidence"
                        ]

                        if confidence > 0:

                            st.metric(
                                "ML Confidence",
                                f"{confidence:.1f}%"
                            )

                        else:

                            st.metric(
                                "ML Confidence",
                                "Not Available"
                            )

                    with col3:

                        st.metric(
                            "Rule-Based Type",
                            result["rule_type"]
                        )

                    # ----------------------------------------
                    # Status
                    # ----------------------------------------

                    st.subheader("Processing Status")

                    if result["status"] == "Processed":

                        st.success(
                            "Processed"
                        )

                    else:

                        st.warning(
                            "Needs Review"
                        )

                    # ----------------------------------------
                    # Invoice information
                    # ----------------------------------------

                    if result["invoice_fields"] is not None:

                        st.header(
                            "Invoice Information"
                        )

                        invoice_fields = result[
                            "invoice_fields"
                        ]

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                "**Invoice Number:**",
                                invoice_fields[
                                    "Invoice Number"
                                ]
                            )

                            st.write(
                                "**Date:**",
                                invoice_fields[
                                    "Date"
                                ]
                            )

                        with col2:

                            st.write(
                                "**Company Name:**",
                                invoice_fields[
                                    "Company Name"
                                ]
                            )

                            st.write(
                                "**Total Amount:**",
                                invoice_fields[
                                    "Total Amount"
                                ]
                            )

                    # ----------------------------------------
                    # Resume information
                    # ----------------------------------------

                    if result["resume_fields"] is not None:

                        st.header(
                            "Resume Information"
                        )

                        resume_fields = result[
                            "resume_fields"
                        ]

                        st.write(
                            "**Name:**",
                            resume_fields["Name"]
                        )

                        st.write(
                            "**Email:**",
                            resume_fields["Email"]
                        )

                        st.write(
                            "**Phone:**",
                            resume_fields["Phone"]
                        )

                        st.write(
                            "**Skills:**",
                            resume_fields["Skills"]
                        )

                    # ----------------------------------------
                    # Processing information
                    # ----------------------------------------

                    st.header(
                        "Processing Information"
                    )

                    st.write(
                        "**Reading method:**",
                        result["reading_method"]
                    )

                    st.write(
                        "**Original file:**",
                        result["file_name"]
                    )

                    st.write(
                        "**Stored file:**",
                        str(relative_file_path)
                    )

                    st.write(
                        "**Document type:**",
                        result["document_type"]
                    )

                    st.write(
                        "**Cleaned text length:**",
                        f"{len(result['cleaned_text'])} characters"
                    )

                    st.write(
                        "**SHA-256:**",
                        file_hash
                    )

                    # ----------------------------------------
                    # Extracted text
                    # ----------------------------------------

                    with st.expander(
                        "View Extracted Text"
                    ):

                        st.text_area(
                            "Cleaned Text",
                            result["cleaned_text"],
                            height=300,
                        )


# ============================================================
# DOCUMENT REPOSITORY
# ============================================================

else:

    st.header("Document Repository")

    st.write(
        "Search, filter, sort, and view saved documents."
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search_term = st.text_input(
        "Search",
        placeholder=(
            "Search filename, company, invoice number, "
            "document type, or text..."
        ),
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        document_type_filter = st.selectbox(
            "Document Type",
            [
                "All",
                "Invoice",
                "Resume",
                "Other",
            ],
        )

    with col2:

        status_filter = st.selectbox(
            "Processing Status",
            [
                "All",
                "Processed",
                "Needs Review",
                "Failed",
            ],
        )

    with col3:

        sort_order = st.selectbox(
            "Sort By",
            [
                "Newest",
                "Oldest",
            ],
        )

    # --------------------------------------------------------
    # Date filters
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        start_date = st.date_input(
            "Upload Date From",
            value=None,
        )

    with col2:

        end_date = st.date_input(
            "Upload Date To",
            value=None,
        )

    # --------------------------------------------------------
    # Clear filters
    # --------------------------------------------------------

    if st.button("Clear Filters"):

        st.rerun()

    # --------------------------------------------------------
    # Convert date values
    # --------------------------------------------------------

    start_date_value = (
        start_date.strftime("%Y-%m-%d")
        if start_date is not None
        else None
    )

    end_date_value = (
        end_date.strftime("%Y-%m-%d")
        if end_date is not None
        else None
    )

    # --------------------------------------------------------
    # Query SQLite repository
    # --------------------------------------------------------

    try:

        documents = filter_documents(
            search_term=search_term,
            document_type=document_type_filter,
            status=status_filter,
            start_date=start_date_value,
            end_date=end_date_value,
            sort_order=sort_order,
        )

    except Exception:

        documents = []

        st.error(
            "The document repository could not be loaded."
        )

    # --------------------------------------------------------
    # Repository results
    # --------------------------------------------------------

    st.write(
        f"Documents found: {len(documents)}"
    )

    if not documents:

        st.info(
            "No documents match the selected search and filters."
        )

    else:

        for document in documents:

            with st.expander(
                f"{document['original_filename']} "
                f"- {document['document_type']} "
                f"- {document['status']}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Document ID:**",
                        document["id"]
                    )

                    st.write(
                        "**Original filename:**",
                        document["original_filename"]
                    )

                    st.write(
                        "**Document type:**",
                        document["document_type"]
                    )

                    st.write(
                        "**Upload date:**",
                        document["upload_date"]
                    )

                    st.write(
                        "**Status:**",
                        document["status"]
                    )

                with col2:

                    st.write(
                        "**Company:**",
                        document["company"] or "Not found"
                    )

                    st.write(
                        "**Invoice number:**",
                        document["invoice_number"] or "Not found"
                    )

                    st.write(
                        "**Total amount:**",
                        document["total_amount"] or "Not found"
                    )

                    st.write(
                        "**File location:**",
                        document["file_path"]
                    )

                # --------------------------------------------
                # Text preview
                # --------------------------------------------

                st.subheader("Text Preview")

                st.text(
                    document["text_preview"]
                    or "No text preview available."
                )

                # --------------------------------------------
                # File access
                # --------------------------------------------

                file_path = (
                    Path(__file__).resolve().parent
                    / document["file_path"]
                )

                if file_path.exists():

                    try:

                        file_data = file_path.read_bytes()

                        st.download_button(
                            "Download / Open File",
                            data=file_data,
                            file_name=document[
                                "original_filename"
                            ],
                            key=f"download_{document['id']}",
                        )

                        extension = file_path.suffix.lower()

                        if extension in [
                            ".jpg",
                            ".jpeg",
                            ".png"
                        ]:

                            st.image(
                                file_data,
                                caption=document[
                                    "original_filename"
                                ],
                            )

                    except Exception:

                        st.warning(
                            "The stored file could not be opened."
                        )

                else:

                    st.warning(
                        "The stored file is no longer available "
                        "at the recorded location."
                    )

                # --------------------------------------------
                # Hash
                # --------------------------------------------

                st.write(
                    "**SHA-256:**",
                    document["file_hash"]
                )

                # --------------------------------------------
                # Full metadata
                # --------------------------------------------

                with st.expander(
                    "View Full Metadata"
                ):

                    st.json(
                        {
                            "id": document["id"],
                            "original_filename": document[
                                "original_filename"
                            ],
                            "stored_filename": document[
                                "stored_filename"
                            ],
                            "document_type": document[
                                "document_type"
                            ],
                            "upload_date": document[
                                "upload_date"
                            ],
                            "company": document[
                                "company"
                            ],
                            "invoice_number": document[
                                "invoice_number"
                            ],
                            "total_amount": document[
                                "total_amount"
                            ],
                            "file_path": document[
                                "file_path"
                            ],
                            "status": document[
                                "status"
                            ],
                        }
                    )
