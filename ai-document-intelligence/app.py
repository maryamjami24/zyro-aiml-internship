import re
import sqlite3
from pathlib import Path

import joblib
import streamlit as st

from database import (
    initialize_database,
    add_document,
    get_document_by_hash,
    filter_documents,
    transition_document,
    get_documents_by_status,
    get_audit_history,
    get_workflow_counts,
    update_document,
)

from workflow import apply_workflow_rules

from storage_manager import (
    initialize_storage,
    calculate_file_hash,
    save_file,
)

from document_processor import (
    extract_text_from_pdf_with_method,
    clean_text,
)

from ocr_processor import (
    extract_text_from_image,
)


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "documents.db"


# ==================================================
# DATABASE COMPATIBILITY
# ==================================================

def ensure_week5_columns():
    """
    Make sure Week 5 fields exist even if the database
    was created with an earlier version of the project.
    """

    columns_to_add = {
        "invoice_date": "TEXT",
    }

    try:
        connection = sqlite3.connect(DATABASE_FILE)
        cursor = connection.cursor()

        cursor.execute("PRAGMA table_info(documents)")
        existing_columns = {
            row[1] for row in cursor.fetchall()
        }

        for column_name, column_type in columns_to_add.items():

            if column_name not in existing_columns:

                cursor.execute(
                    f"ALTER TABLE documents "
                    f"ADD COLUMN {column_name} {column_type}"
                )

        connection.commit()
        connection.close()

    except Exception:
        pass


# ==================================================
# LOAD ML MODEL
# ==================================================

MODEL_FILE = BASE_DIR / "document_classifier.pkl"

classifier_model = None

if MODEL_FILE.exists():

    try:
        classifier_model = joblib.load(MODEL_FILE)

    except Exception:
        classifier_model = None


# ==================================================
# TEXT CLEANING
# ==================================================

def normalize_spaces(text):

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


# ==================================================
# RULE-BASED CLASSIFICATION
# ==================================================

def classify_document_rule_based(text):

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
        "grand total",
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
        "achievements",
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

    if (
        invoice_score >= 2
        and invoice_score > resume_score
    ):
        return "Invoice"

    if (
        resume_score >= 2
        and resume_score > invoice_score
    ):
        return "Resume"

    return "Other"


# ==================================================
# ML CLASSIFICATION
# ==================================================

def classify_document_ml(text):

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

            probabilities = (
                classifier_model.predict_proba(
                    [text]
                )[0]
            )

            confidence = (
                max(probabilities) * 100
            )

        return document_type, confidence

    except Exception:

        return "Other", 0.0


# ==================================================
# SECTION FINDER
# ==================================================

def find_section(
    lines,
    heading_patterns,
    stop_patterns
):

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
            section_lines.append(
                line.strip()
            )

    return section_lines


# ==================================================
# RESUME NAME
# ==================================================

def extract_resume_name(lines):

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
        "hobbies",
    }

    ignored_phrases = [
        "resume template",
        "build this resume",
        "make this resume",
        "download",
        "linkedin",
        "pinterest",
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


# ==================================================
# RESUME EXTRACTION
# ==================================================

def extract_resume_fields(text):

    fields = {
        "Name": "Not Found",
        "Email": "Not Found",
        "Phone": "Not Found",
        "Skills": "Not Found",
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
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        normalized_text,
    )

    if email_match:
        fields["Email"] = email_match.group(0)

    phone_patterns = [
        r"\+\d{1,3}[\s\-()]?\d{2,4}[\s\-()]?\d{3,4}[\s\-()]?\d{3,4}",
        r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",
        r"\b\d{3}\s\d{3}\s\d{4}\b",
        r"\b\d{10,15}\b",
    ]

    for pattern in phone_patterns:

        phone_match = re.search(
            pattern,
            normalized_text
        )

        if phone_match:

            fields["Phone"] = (
                phone_match.group(0)
            )

            break

    skill_headings = [
        r"skills",
        r"key skills",
        r"core skills",
        r"technical skills",
        r"professional skills",
        r"competencies",
        r"core competencies",
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
        r"links",
    ]

    skill_lines = find_section(
        lines,
        skill_headings,
        stop_headings
    )

    cleaned_skills = []

    ignored_skill_text = [
        "resume templates",
        "build this resume",
        "make this resume",
        "linkedin",
        "pinterest",
    ]

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

        if (
            skill.lower()
            in ignored_skill_text
        ):
            continue

        if len(skill.split()) > 15:
            continue

        if (
            skill.lower()
            not in [
                item.lower()
                for item in cleaned_skills
            ]
        ):

            cleaned_skills.append(skill)

    if cleaned_skills:

        fields["Skills"] = ", ".join(
            cleaned_skills
        )

    return fields


# ==================================================
# INVOICE NUMBER
# ==================================================

def extract_invoice_number(text):

    if not text or not text.strip():
        return "Not Found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    same_line_patterns = [

        r"^invoice\s*(?:number|no\.?|#)\s*[:\-]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9./_-]{2,})$",

        r"^proforma\s+invoice\s*#?\s*[:\-]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9./_-]{2,})$",
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
                    "date",
                ]:

                    return value

    standalone_patterns = [

        r"(?<![\d-])\d{4,10}-\d{2,10}/\d{1,4}(?![\d/])",

        r"(?<![\d/])\d{4,10}/\d{1,4}(?![\d/])",

        r"\bINV[\s_-]?\d[\w./_-]*\b",
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


# ==================================================
# INVOICE DATE
# ==================================================

def extract_invoice_date(text):

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

        r"\b[A-Za-z]+\s+\d{1,2},\s+\d{4}\b",
    ]

    labeled_patterns = [

        r"^(?:invoice\s*date|date\s*of\s*issue)"
        r"\s*[:\-]\s*"
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})$",

        r"^(?:invoice\s*date|date\s*of\s*issue)"
        r"\s*[:\-]\s*"
        r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})$",
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
        "date:",
    ]

    for index, line in enumerate(lines):

        clean_line = line.lower().strip()

        if clean_line in date_labels:

            for candidate in lines[
                index + 1:index + 12
            ]:

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


# ==================================================
# COMPANY NAME
# ==================================================

def extract_company_name(text):

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

        r"^billed\s*by\s*[:\-]\s*(.+)$",
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
        "john doe",
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
            "new york",
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


# ==================================================
# TOTAL AMOUNT
# ==================================================

def extract_total_amount(text):

    if not text or not text.strip():
        return "Not Found"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    amount_pattern = (
        r"(?<![\d.])"
        r"(?:[$€£₹]\s*)?"
        r"\d+(?:,\d{3})*\.\d{2}"
        r"\s*(?:[$€£₹])?"
        r"(?![\d.])"
    )

    priority_labels = [
        "grand total",
        "total amount",
        "total due",
        "balance due",
        "amount due",
    ]

    for index, line in enumerate(lines):

        clean_line = line.lower().strip()

        for label in priority_labels:

            pattern = (
                rf"^{re.escape(label)}\s*[:\-]?"
                rf"\s*({amount_pattern})$"
            )

            match = re.search(
                pattern,
                line,
                re.IGNORECASE
            )

            if match:
                return match.group(1).strip()

        if clean_line in priority_labels:

            for candidate in lines[
                index + 1:index + 6
            ]:

                match = re.search(
                    amount_pattern,
                    candidate
                )

                if match:
                    return match.group(0).strip()

    for index, line in enumerate(lines):

        clean_line = line.lower().strip()

        if clean_line == "total":

            for candidate in lines[
                index + 1:index + 6
            ]:

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
            re.IGNORECASE,
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


# ==================================================
# INVOICE EXTRACTION
# ==================================================

def extract_invoice_fields(text):

    fields = {
        "Invoice Number": "Not Found",
        "Date": "Not Found",
        "Company Name": "Not Found",
        "Total Amount": "Not Found",
    }

    if not text or not text.strip():
        return fields

    normalized_text = normalize_spaces(text)

    fields["Invoice Number"] = (
        extract_invoice_number(
            normalized_text
        )
    )

    fields["Date"] = (
        extract_invoice_date(
            normalized_text
        )
    )

    fields["Company Name"] = (
        extract_company_name(
            normalized_text
        )
    )

    fields["Total Amount"] = (
        extract_total_amount(
            normalized_text
        )
    )

    return fields


# ==================================================
# DOCUMENT PROCESSING
# ==================================================

def process_document(uploaded_file):

    file_name = uploaded_file.name

    extension = Path(file_name).suffix.lower()

    temp_path = (
        BASE_DIR
        / f"temp_uploaded_document{extension}"
    )

    try:

        with open(temp_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )

        if extension == ".pdf":

            (
                raw_text,
                reading_method,
            ) = extract_text_from_pdf_with_method(
                str(temp_path)
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

        elif extension in [
            ".jpg",
            ".jpeg",
            ".png",
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
                ),
            }

        cleaned_text = clean_text(
            raw_text
        )

        if not cleaned_text:

            return {
                "success": False,
                "error": (
                    "No readable text was found "
                    "in this document."
                ),
            }

        if len(cleaned_text.strip()) < 20:

            return {
                "success": False,
                "error": (
                    "The extracted text is too short "
                    "to identify this document reliably."
                ),
            }

        ml_type, ml_confidence = (
            classify_document_ml(
                cleaned_text
            )
        )

        rule_type = (
            classify_document_rule_based(
                cleaned_text
            )
        )

        if (
            ml_type in ["Invoice", "Resume"]
            and ml_type == rule_type
        ):

            document_type = ml_type

        elif rule_type in [
            "Invoice",
            "Resume",
        ]:

            document_type = rule_type

        else:

            document_type = "Other"

        invoice_fields = None
        resume_fields = None

        if document_type == "Invoice":

            invoice_fields = (
                extract_invoice_fields(
                    cleaned_text
                )
            )

        elif document_type == "Resume":

            resume_fields = (
                extract_resume_fields(
                    cleaned_text
                )
            )

        # ------------------------------------------
        # WEEK 5 WORKFLOW
        # ------------------------------------------

        workflow_data = {}

        if (
            document_type == "Invoice"
            and invoice_fields
        ):

            workflow_data = {
                "invoice_number": (
                    invoice_fields.get(
                        "Invoice Number"
                    )
                ),
                "invoice_date": (
                    invoice_fields.get(
                        "Date"
                    )
                ),
                "company": (
                    invoice_fields.get(
                        "Company Name"
                    )
                ),
                "total_amount": (
                    invoice_fields.get(
                        "Total Amount"
                    )
                ),
            }

        elif (
            document_type == "Resume"
            and resume_fields
        ):

            workflow_data = {
                "person_name": (
                    resume_fields.get(
                        "Name"
                    )
                ),
                "email": (
                    resume_fields.get(
                        "Email"
                    )
                ),
                "phone": (
                    resume_fields.get(
                        "Phone"
                    )
                ),
                "skills": (
                    resume_fields.get(
                        "Skills"
                    )
                ),
            }

        workflow_result = (
            apply_workflow_rules(
                document_type=document_type,
                extracted_data=workflow_data,
                confidence=ml_confidence,
            )
        )

        workflow_decision = (
            workflow_result["decision"]
        )

        workflow_reason = (
            workflow_result["reason"]
        )

        if workflow_decision == "Approved":

            status = "Approved"

        else:

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
            "status": status,
            "workflow_decision": workflow_decision,
            "workflow_reason": workflow_reason,
            "validation_status": (
                workflow_result[
                    "validation"
                ]["status"]
            ),
            "validation_errors": (
                workflow_result[
                    "validation"
                ]["errors"]
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "error": (
                "The document could not be processed. "
                f"Details: {error}"
            ),
        }

    finally:

        try:

            if temp_path.exists():
                temp_path.unlink()

        except Exception:
            pass


# ==================================================
# SAVE WORKFLOW DOCUMENT
# ==================================================

def save_processed_document(
    result,
    file_bytes,
    file_hash,
):

    stored_filename = None
    file_path = None

    try:

        (
            stored_filename,
            file_path,
        ) = save_file(
            file_bytes,
            result["file_name"],
            result["document_type"],
        )

        relative_file_path = (
            file_path.relative_to(
                BASE_DIR
            )
        )

        company = None
        invoice_number = None
        total_amount = None
        invoice_date = None

        person_name = None
        email = None
        phone = None
        skills = None

        if result["invoice_fields"]:

            fields = result[
                "invoice_fields"
            ]

            company = fields.get(
                "Company Name"
            )

            invoice_number = fields.get(
                "Invoice Number"
            )

            total_amount = fields.get(
                "Total Amount"
            )

            invoice_date = fields.get(
                "Date"
            )

        if result["resume_fields"]:

            fields = result[
                "resume_fields"
            ]

            person_name = fields.get(
                "Name"
            )

            email = fields.get(
                "Email"
            )

            phone = fields.get(
                "Phone"
            )

            skills = fields.get(
                "Skills"
            )

        # First create the record in New state.
        document_id = add_document(
            original_filename=(
                result["file_name"]
            ),
            stored_filename=(
                stored_filename
            ),
            document_type=(
                result["document_type"]
            ),
            company=company,
            invoice_number=(
                invoice_number
            ),
            total_amount=(
                total_amount
            ),
            file_path=(
                str(relative_file_path)
            ),
            text_preview=(
                result["cleaned_text"][:500]
            ),
            file_hash=file_hash,
            status="New",
        )

        # Move New -> Processing.
        try:

            transition_document(
                document_id,
                "Processing",
                reason="Document processing started.",
            )

        except Exception:
            pass

        # Store Week 5 metadata.
        try:

            update_document(
                document_id,
                predicted_type=(
                    result["ml_type"]
                ),
                confidence=(
                    result["ml_confidence"]
                ),
                review_reason=(
                    result["workflow_reason"]
                ),
                validation_status=(
                    result["validation_status"]
                ),
                validation_errors=(
                    " | ".join(
                        result[
                            "validation_errors"
                        ]
                    )
                ),
                person_name=person_name,
                email=email,
                phone=phone,
                skills=skills,
                invoice_date=invoice_date,
            )

        except TypeError:

            # Compatibility fallback for databases
            # that do not yet accept invoice_date.
            update_document(
                document_id,
                predicted_type=(
                    result["ml_type"]
                ),
                confidence=(
                    result["ml_confidence"]
                ),
                review_reason=(
                    result["workflow_reason"]
                ),
                validation_status=(
                    result["validation_status"]
                ),
                validation_errors=(
                    " | ".join(
                        result[
                            "validation_errors"
                        ]
                    )
                ),
                person_name=person_name,
                email=email,
                phone=phone,
                skills=skills,
            )

        # Apply final workflow decision.
        final_status = (
            result["status"]
        )

        try:

            transition_document(
                document_id,
                final_status,
                reason=(
                    result[
                        "workflow_reason"
                    ]
                ),
            )

        except Exception:

            try:

                update_document(
                    document_id,
                    status=final_status,
                )

            except Exception:
                pass

        return document_id, file_path

    except Exception:

        try:

            if file_path and file_path.exists():
                file_path.unlink()

        except Exception:
            pass

        raise


# ==================================================
# WORKFLOW STATUS BADGE
# ==================================================

def show_status(status):

    if status == "New":

        st.info("New")

    elif status == "Processing":

        st.info("Processing")

    elif status == "Needs Review":

        st.warning("Needs Review")

    elif status == "Approved":

        st.success("Approved")

    elif status == "Rejected":

        st.error("Rejected")

    elif status == "Completed":

        st.success("Completed")

    else:

        st.write(status)


# ==================================================
# DOCUMENT REVIEW
# ==================================================

def render_review_document(document):

    document = dict(document)
    document_id = document["id"]

    st.subheader(
        document["original_filename"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            "**Document ID:**",
            document_id
        )

        st.write(
            "**Type:**",
            document["document_type"]
        )

    with col2:

        st.write(
            "**Status:**",
            document["status"]
        )

        st.write(
            "**Upload Date:**",
            document["upload_date"]
        )

    with col3:

        confidence = document["confidence"] if "confidence" in document.keys() else None

        if confidence is not None and float(
            confidence or 0
        ) > 0:

            st.write(
                "**ML Confidence:**",
                f"{float(confidence):.1f}%"
            )

        else:

            st.write(
                "**ML Confidence:**",
                "Not Available"
            )

    st.divider()

    st.write(
        "**Workflow Reason:**"
    )

    st.warning(
        document.get(
            "review_reason"
        )
        or "No review reason recorded."
    )

    st.write(
        "**Validation Status:**",
        document.get(
            "validation_status"
        )
        or "Not recorded"
    )

    validation_errors = document.get(
        "validation_errors"
    )

    if validation_errors:

        st.write(
            "**Validation Errors:**"
        )

        for error in str(
            validation_errors
        ).split(" | "):

            if error.strip():

                st.error(error)

    st.divider()

    if document["document_type"] == "Invoice":

        st.subheader(
            "Extracted Invoice Fields"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Invoice Number:**",
                document.get(
                    "invoice_number"
                )
                or "Not Found"
            )

            st.write(
                "**Date:**",
                document.get(
                    "invoice_date"
                )
                or "Not Found"
            )

        with col2:

            st.write(
                "**Company:**",
                document.get(
                    "company"
                )
                or "Not Found"
            )

            st.write(
                "**Total Amount:**",
                document.get(
                    "total_amount"
                )
                or "Not Found"
            )

    elif document["document_type"] == "Resume":

        st.subheader(
            "Extracted Resume Fields"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Name:**",
                document.get(
                    "person_name"
                )
                or "Not Found"
            )

            st.write(
                "**Email:**",
                document.get(
                    "email"
                )
                or "Not Found"
            )

        with col2:

            st.write(
                "**Phone:**",
                document.get(
                    "phone"
                )
                or "Not Found"
            )

            st.write(
                "**Skills:**",
                document.get(
                    "skills"
                )
                or "Not Found"
            )

    st.divider()

    st.subheader(
        "Review Decision"
    )

    if document["status"] == "Needs Review":

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Approve",
                key=f"approve_{document_id}",
                use_container_width=True,
            ):

                try:

                    transition_document(
                        document_id,
                        "Approved",
                        reason=(
                            "Human reviewer approved "
                            "the document."
                        ),
                    )

                    st.success(
                        "Document approved."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Approval failed: {error}"
                    )

        with col2:

            reject_reason = st.text_input(
                "Rejection reason",
                key=f"reject_reason_{document_id}",
                placeholder=(
                    "Enter a short rejection reason"
                ),
            )

            if st.button(
                "Reject",
                key=f"reject_{document_id}",
                use_container_width=True,
            ):

                if not reject_reason.strip():

                    st.error(
                        "A rejection reason is required."
                    )

                else:

                    try:

                        transition_document(
                            document_id,
                            "Rejected",
                            reason=(
                                reject_reason.strip()
                            ),
                        )

                        st.success(
                            "Document rejected."
                        )

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Rejection failed: {error}"
                        )

    elif document["status"] == "Approved":

        if st.button(
            "Mark as Completed",
            key=f"complete_{document_id}",
            use_container_width=True,
        ):

            try:

                transition_document(
                    document_id,
                    "Completed",
                    reason=(
                        "Document workflow completed."
                    ),
                )

                st.success(
                    "Document marked as Completed."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Completion failed: {error}"
                )

    elif document["status"] == "Rejected":

        if st.button(
            "Mark Rejected Document Completed",
            key=f"complete_rejected_{document_id}",
            use_container_width=True,
        ):

            try:

                transition_document(
                    document_id,
                    "Completed",
                    reason=(
                        "Rejected document workflow closed."
                    ),
                )

                st.success(
                    "Workflow completed."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Completion failed: {error}"
                )


# ==================================================
# BATCH PROCESSING
# ==================================================

def process_existing_document(document):

    """
    Re-apply Week 5 validation/workflow rules to an
    already stored document.

    The function returns an individual result so one
    failed document does not stop the batch.
    """

    try:

        document_type = (
            document["document_type"]
        )

        data = {}

        if document_type == "Invoice":

            data = {
                "invoice_number": (
                    document.get(
                        "invoice_number"
                    )
                ),
                "invoice_date": (
                    document.get(
                        "invoice_date"
                    )
                ),
                "company": (
                    document.get(
                        "company"
                    )
                ),
                "total_amount": (
                    document.get(
                        "total_amount"
                    )
                ),
            }

        elif document_type == "Resume":

            data = {
                "person_name": (
                    document.get(
                        "person_name"
                    )
                ),
                "email": (
                    document.get(
                        "email"
                    )
                ),
                "phone": (
                    document.get(
                        "phone"
                    )
                ),
                "skills": (
                    document.get(
                        "skills"
                    )
                ),
            }

        result = apply_workflow_rules(
            document_type=document_type,
            extracted_data=data,
            confidence=document.get(
                "confidence"
            ),
        )

        decision = result["decision"]

        current_status = document[
            "status"
        ]

        # Rejected/Completed are terminal for the
        # normal workflow, so report instead of forcing
        # an invalid transition.
        if current_status in [
            "Completed",
        ]:

            return {
                "id": document["id"],
                "filename": document[
                    "original_filename"
                ],
                "result": "Skipped",
                "reason": (
                    "Document is already Completed."
                ),
            }

        target_status = decision

        if current_status == target_status:

            return {
                "id": document["id"],
                "filename": document[
                    "original_filename"
                ],
                "result": target_status,
                "reason": result["reason"],
            }

        if current_status == "New":

            transition_document(
                document["id"],
                "Processing",
                reason="Batch processing started.",
            )

            current_status = "Processing"

        if target_status == "Approved":

            transition_document(
                document["id"],
                "Approved",
                reason=result["reason"],
            )

        elif target_status == "Needs Review":

            transition_document(
                document["id"],
                "Needs Review",
                reason=result["reason"],
            )

        else:

            return {
                "id": document["id"],
                "filename": document[
                    "original_filename"
                ],
                "result": "Failed",
                "reason": (
                    "Unsupported workflow decision."
                ),
            }

        update_document(
            document["id"],
            review_reason=result["reason"],
            validation_status=(
                result["validation"]["status"]
            ),
            validation_errors=(
                " | ".join(
                    result["validation"]["errors"]
                )
            ),
        )

        return {
            "id": document["id"],
            "filename": document[
                "original_filename"
            ],
            "result": target_status,
            "reason": result["reason"],
        }

    except Exception as error:

        return {
            "id": document["id"],
            "filename": document[
                "original_filename"
            ],
            "result": "Failed",
            "reason": str(error),
        }


# ==================================================
# INITIALIZATION
# ==================================================

initialize_database()
initialize_storage()
ensure_week5_columns()


MAX_FILE_SIZE = 10 * 1024 * 1024

SUPPORTED_EXTENSIONS = [
    "pdf",
    "jpg",
    "jpeg",
    "png",
]


# ==================================================
# MAIN HEADER
# ==================================================

st.title(
    "AI Document Intelligence & Workflow Platform"
)

st.write(
    "Upload, process, validate, review, "
    "approve, reject, and manage documents."
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "Document Management"
)

st.sidebar.caption(
    "AI document processing and workflow platform"
)

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Upload Document",
        "Human Review Queue",
        "Batch Workflow",
        "Document Repository",
        "Audit History",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Supported: PDF, JPG, JPEG, PNG"
)

st.sidebar.caption(
    "Maximum file size: 10 MB"
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header(
        "Workflow Dashboard"
    )

    st.write(
        "Overview of document processing "
        "and workflow activity."
    )

    try:

        counts = get_workflow_counts()

    except Exception:

        counts = {}

        st.error(
            "Workflow metrics could not be loaded."
        )

    total = counts.get(
        "total",
        0
    )

    processed = counts.get(
        "processed",
        0
    )

    needs_review = counts.get(
        "needs_review",
        0
    )

    approved = counts.get(
        "approved",
        0
    )

    rejected = counts.get(
        "rejected",
        0
    )

    completed = counts.get(
        "completed",
        0
    )

    failed = counts.get(
        "failed",
        0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Documents",
            total
        )

    with col2:

        st.metric(
            "Processed",
            processed
        )

    with col3:

        st.metric(
            "Needs Review",
            needs_review
        )

    with col4:

        st.metric(
            "Completed",
            completed
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Approved",
            approved
        )

    with col2:

        st.metric(
            "Rejected",
            rejected
        )

    with col3:

        st.metric(
            "Failed",
            failed
        )

    st.divider()

    st.subheader(
        "Documents by Type"
    )

    by_type = counts.get(
        "by_type",
        {}
    )

    if by_type:

        type_cols = st.columns(
            len(by_type)
        )

        for column, (
            document_type,
            count
        ) in zip(
            type_cols,
            by_type.items()
        ):

            with column:

                st.metric(
                    document_type,
                    count
                )

    else:

        st.info(
            "No document type data available yet."
        )

    st.divider()

    st.subheader(
        "Workflow States"
    )

    workflow_data = {
        "New": counts.get("new", 0),
        "Processing": counts.get("processing", 0),
        "Needs Review": needs_review,
        "Approved": approved,
        "Rejected": rejected,
        "Completed": completed,
    }

    for state, count in workflow_data.items():

        st.write(
            f"**{state}:** {count}"
        )


# ==================================================
# UPLOAD PAGE
# ==================================================

elif page == "Upload Document":

    st.header(
        "Upload Document"
    )

    st.write(
        "Upload a document to extract information, "
        "classify it, validate it, and apply workflow rules."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=SUPPORTED_EXTENSIONS,
        max_upload_size=10,
        help=(
            "Supported formats: PDF, JPG, JPEG, PNG. "
            "Maximum size: 10 MB."
        ),
    )

    if uploaded_file is None:

        st.info(
            "Select a document above to begin processing."
        )

    else:

        file_bytes = (
            uploaded_file.getvalue()
        )

        file_size_mb = (
            len(file_bytes)
            / (1024 * 1024)
        )

        st.caption(
            f"Selected file: {uploaded_file.name} "
            f"({file_size_mb:.2f} MB)"
        )

        if len(file_bytes) > MAX_FILE_SIZE:

            st.error(
                "File is too large. "
                "Maximum allowed size is 10 MB."
            )

        else:

            file_hash = (
                calculate_file_hash(
                    file_bytes
                )
            )

            existing_document = (
                get_document_by_hash(
                    file_hash
                )
            )

            if existing_document is not None:

                st.warning(
                    "Duplicate document detected."
                )

                st.write(
                    "This file already exists "
                    "in the repository."
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        "**Filename**"
                    )

                    st.write(
                        existing_document[
                            "original_filename"
                        ]
                    )

                with col2:

                    st.write(
                        "**Document Type**"
                    )

                    st.write(
                        existing_document[
                            "document_type"
                        ]
                    )

                with col3:

                    st.write(
                        "**Status**"
                    )

                    show_status(
                        existing_document[
                            "status"
                        ]
                    )

                st.info(
                    "No new database record or "
                    "file was created."
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

                    try:

                        (
                            document_id,
                            file_path,
                        ) = save_processed_document(
                            result,
                            file_bytes,
                            file_hash,
                        )

                        st.success(
                            f"Document saved successfully. "
                            f"Document ID: {document_id}"
                        )

                    except Exception as error:

                        st.error(
                            "The document could not "
                            "be stored."
                        )

                        st.code(
                            str(error)
                        )

                        document_id = None
                        file_path = None

                    # ----------------------------------
                    # ANALYSIS
                    # ----------------------------------

                    st.divider()

                    st.header(
                        "Document Analysis"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "ML Type",
                            result["ml_type"]
                        )

                    with col2:

                        confidence = (
                            result[
                                "ml_confidence"
                            ]
                        )

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

                    st.subheader(
                        "Workflow Result"
                    )

                    st.write(
                        "**Document Type:**",
                        result["document_type"]
                    )

                    st.write(
                        "**Status:**"
                    )

                    show_status(
                        result["status"]
                    )

                    st.write(
                        "**Workflow Decision:**",
                        result[
                            "workflow_decision"
                        ]
                    )

                    st.write(
                        "**Reason:**",
                        result[
                            "workflow_reason"
                        ]
                    )

                    st.write(
                        "**Validation Status:**",
                        result[
                            "validation_status"
                        ]
                    )

                    if result[
                        "validation_errors"
                    ]:

                        st.subheader(
                            "Validation Errors"
                        )

                        for error in result[
                            "validation_errors"
                        ]:

                            st.error(error)

                    # ----------------------------------
                    # INVOICE
                    # ----------------------------------

                    if (
                        result[
                            "invoice_fields"
                        ] is not None
                    ):

                        st.divider()

                        st.subheader(
                            "Invoice Information"
                        )

                        fields = result[
                            "invoice_fields"
                        ]

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                "**Invoice Number**"
                            )

                            st.write(
                                fields[
                                    "Invoice Number"
                                ]
                            )

                            st.write(
                                "**Date**"
                            )

                            st.write(
                                fields[
                                    "Date"
                                ]
                            )

                        with col2:

                            st.write(
                                "**Company Name**"
                            )

                            st.write(
                                fields[
                                    "Company Name"
                                ]
                            )

                            st.write(
                                "**Total Amount**"
                            )

                            st.write(
                                fields[
                                    "Total Amount"
                                ]
                            )

                    # ----------------------------------
                    # RESUME
                    # ----------------------------------

                    if (
                        result[
                            "resume_fields"
                        ] is not None
                    ):

                        st.divider()

                        st.subheader(
                            "Resume Information"
                        )

                        fields = result[
                            "resume_fields"
                        ]

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                "**Name**"
                            )

                            st.write(
                                fields[
                                    "Name"
                                ]
                            )

                            st.write(
                                "**Email**"
                            )

                            st.write(
                                fields[
                                    "Email"
                                ]
                            )

                        with col2:

                            st.write(
                                "**Phone**"
                            )

                            st.write(
                                fields[
                                    "Phone"
                                ]
                            )

                            st.write(
                                "**Skills**"
                            )

                            st.write(
                                fields[
                                    "Skills"
                                ]
                            )

                    # ----------------------------------
                    # PROCESSING INFORMATION
                    # ----------------------------------

                    st.divider()

                    st.subheader(
                        "Processing Information"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**Reading Method:**",
                            result[
                                "reading_method"
                            ]
                        )

                        st.write(
                            "**Original File:**",
                            result[
                                "file_name"
                            ]
                        )

                        st.write(
                            "**Document Type:**",
                            result[
                                "document_type"
                            ]
                        )

                    with col2:

                        if file_path:

                            try:

                                relative_path = (
                                    file_path.relative_to(
                                        BASE_DIR
                                    )
                                )

                                st.write(
                                    "**Stored File:**",
                                    str(
                                        relative_path
                                    )
                                )

                            except Exception:

                                st.write(
                                    "**Stored File:**",
                                    str(file_path)
                                )

                        st.write(
                            "**Cleaned Text Length:**",
                            f"{len(result['cleaned_text'])} characters"
                        )

                    st.write(
                        "**SHA-256 Hash:**"
                    )

                    st.code(
                        file_hash,
                        language=None
                    )

                    with st.expander(
                        "View Extracted Text"
                    ):

                        st.text_area(
                            "Cleaned Text",
                            result[
                                "cleaned_text"
                            ],
                            height=300
                        )


# ==================================================
# HUMAN REVIEW QUEUE
# ==================================================

elif page == "Human Review Queue":

    st.header(
        "Human Review Queue"
    )

    st.write(
        "Documents requiring human attention "
        "because validation failed or confidence "
        "was below the workflow threshold."
    )

    st.divider()

    try:

        review_documents = (
            get_documents_by_status(
                "Needs Review"
            )
        )

    except Exception as error:

        review_documents = []

        st.error(
            f"Review queue could not be loaded: {error}"
        )

    st.metric(
        "Documents Requiring Review",
        len(review_documents)
    )

    st.divider()

    if not review_documents:

        st.success(
            "No documents are currently waiting for review."
        )

    else:

        for document in review_documents:

            with st.expander(
                f"{document['original_filename']} "
                f"• {document['document_type']}"
            ):

                render_review_document(
                    document
                )


# ==================================================
# BATCH WORKFLOW
# ==================================================

elif page == "Batch Workflow":

    st.header(
        "Batch Workflow"
    )

    st.write(
        "Select multiple stored documents and apply "
        "the same validation and workflow rules."
    )

    st.divider()

    try:

        all_documents = filter_documents(
            search_term="",
            document_type="All",
            status="All",
            start_date=None,
            end_date=None,
            sort_order="Newest",
        )

    except Exception as error:

        all_documents = []

        st.error(
            f"Documents could not be loaded: {error}"
        )

    if not all_documents:

        st.info(
            "No stored documents are available "
            "for batch processing."
        )

    else:
        all_documents = [dict(document) for document in all_documents]

        document_options = {
            (
                f"{document['id']} — "
                f"{document['original_filename']} "
                f"({document['status']})"
            ): document["id"]
            for document in all_documents
        }

        selected_labels = st.multiselect(
            "Select documents",
            list(document_options.keys())
        )

        selected_ids = [
            document_options[label]
            for label in selected_labels
        ]

        if st.button(
            "Run Batch Workflow",
            type="primary",
            use_container_width=True,
        ):

            if not selected_ids:

                st.warning(
                    "Select at least one document."
                )

            else:

                results = []

                progress = st.progress(0)

                total_selected = len(
                    selected_ids
                )

                for index, document_id in enumerate(
                    selected_ids
                ):

                    document = next(
                        (
                            item
                            for item in all_documents
                            if item["id"] == document_id
                        ),
                        None,
                    )

                    if document is None:

                        results.append(
                            {
                                "id": document_id,
                                "filename": "Unknown",
                                "result": "Failed",
                                "reason": (
                                    "Document was not found."
                                ),
                            }
                        )

                    else:

                        results.append(
                            process_existing_document(
                                document
                            )
                        )

                    progress.progress(
                        (index + 1)
                        / total_selected
                    )

                st.divider()

                processed_count = sum(
                    1
                    for item in results
                    if item["result"]
                    in [
                        "Approved",
                        "Completed",
                    ]
                )

                review_count = sum(
                    1
                    for item in results
                    if item["result"]
                    == "Needs Review"
                )

                failed_count = sum(
                    1
                    for item in results
                    if item["result"]
                    == "Failed"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Processed",
                        processed_count
                    )

                with col2:

                    st.metric(
                        "Needs Review",
                        review_count
                    )

                with col3:

                    st.metric(
                        "Failed",
                        failed_count
                    )

                st.subheader(
                    "Individual Results"
                )

                for item in results:

                    if item["result"] == "Failed":

                        st.error(
                            f"{item['filename']} — "
                            f"{item['result']}: "
                            f"{item['reason']}"
                        )

                    elif (
                        item["result"]
                        == "Needs Review"
                    ):

                        st.warning(
                            f"{item['filename']} — "
                            f"{item['result']}: "
                            f"{item['reason']}"
                        )

                    else:

                        st.success(
                            f"{item['filename']} — "
                            f"{item['result']}: "
                            f"{item['reason']}"
                        )


# ==================================================
# DOCUMENT REPOSITORY
# ==================================================

elif page == "Document Repository":

    st.header(
        "Document Repository"
    )

    st.write(
        "Search, filter, sort, and manage "
        "stored documents and workflow states."
    )

    st.divider()

    st.subheader(
        "Search and Workflow Filters"
    )

    search_term = st.text_input(
        "Search Documents",
        placeholder=(
            "Filename, company, invoice number, "
            "document type, name, email, or skills..."
        ),
    )

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
            "Workflow Status",
            [
                "All",
                "New",
                "Processing",
                "Needs Review",
                "Approved",
                "Rejected",
                "Completed",
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

    if st.button(
        "Clear Filters",
        use_container_width=True,
    ):

        st.rerun()

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

    try:

        documents = filter_documents(
            search_term=search_term,
            document_type=(
                document_type_filter
            ),
            status=status_filter,
            start_date=start_date_value,
            end_date=end_date_value,
            sort_order=sort_order,
        )

    except Exception as error:

        documents = []

        st.error(
            f"The document repository "
            f"could not be loaded: {error}"
        )

    st.divider()

    st.write(
        f"**Documents found:** {len(documents)}"
    )

    if not documents:

        st.info(
            "No documents match the "
            "selected search and filters."
        )

    else:

        for document in documents:
            document = dict(document)

            document_title = (
                f"{document['original_filename']} "
                f"• {document['document_type']} "
                f"• {document['status']}"
            )

            with st.expander(
                document_title
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Document ID:**",
                        document["id"]
                    )

                    st.write(
                        "**Original Filename:**",
                        document[
                            "original_filename"
                        ]
                    )

                    st.write(
                        "**Document Type:**",
                        document[
                            "document_type"
                        ]
                    )

                    st.write(
                        "**Upload Date:**",
                        document[
                            "upload_date"
                        ]
                    )

                    st.write(
                        "**Workflow Status:**"
                    )

                    show_status(
                        document[
                            "status"
                        ]
                    )

                with col2:

                    st.write(
                        "**Company:**",
                        (
                            document["company"]
                            if "company" in document.keys()
                            else None
                        )
                        or "Not found"
                    )

                    st.write(
                        "**Invoice Number:**",
                        document.get(
                            "invoice_number"
                        )
                        or "Not found"
                    )

                    st.write(
                        "**Invoice Date:**",
                        document.get(
                            "invoice_date"
                        )
                        or "Not found"
                    )

                    st.write(
                        "**Total Amount:**",
                        document.get(
                            "total_amount"
                        )
                        or "Not found"
                    )

                st.divider()

                st.subheader(
                    "Workflow Information"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Predicted Type:**",
                        document.get(
                            "predicted_type"
                        )
                        or "Not available"
                    )

                    confidence = document.get(
                        "confidence"
                    )

                    if confidence is not None and float(
                        confidence or 0
                    ) > 0:

                        st.write(
                            "**Confidence:**",
                            f"{float(confidence):.1f}%"
                        )

                    else:

                        st.write(
                            "**Confidence:**",
                            "Not Available"
                        )

                with col2:

                    st.write(
                        "**Validation:**",
                        document.get(
                            "validation_status"
                        )
                        or "Not recorded"
                    )

                    st.write(
                        "**Review Reason:**",
                        document.get(
                            "review_reason"
                        )
                        or "None"
                    )

                validation_errors = document.get(
                    "validation_errors"
                )

                if validation_errors:

                    st.write(
                        "**Validation Errors:**"
                    )

                    for error in str(
                        validation_errors
                    ).split(" | "):

                        if error.strip():

                            st.error(error)

                st.divider()

                st.subheader(
                    "Text Preview"
                )

                st.text(
                    document.get(
                        "text_preview"
                    )
                    or "No text preview available."
                )

                st.subheader(
                    "File Access"
                )

                file_path = (
                    BASE_DIR
                    / document[
                        "file_path"
                    ]
                )

                if file_path.exists():

                    try:

                        file_data = (
                            file_path.read_bytes()
                        )

                        st.download_button(
                            "Download / Open File",
                            data=file_data,
                            file_name=(
                                document[
                                    "original_filename"
                                ]
                            ),
                            key=(
                                f"download_"
                                f"{document['id']}"
                            ),
                            use_container_width=True,
                        )

                        extension = (
                            file_path.suffix.lower()
                        )

                        if extension in [
                            ".jpg",
                            ".jpeg",
                            ".png",
                        ]:

                            st.image(
                                file_data,
                                caption=(
                                    document[
                                        "original_filename"
                                    ]
                                ),
                            )

                    except Exception:

                        st.warning(
                            "The stored file "
                            "could not be opened."
                        )

                else:

                    st.warning(
                        "The stored file is no longer "
                        "available at the recorded location."
                    )

                st.write(
                    "**SHA-256 Hash:**"
                )

                st.code(
                    document["file_hash"],
                    language=None
                )

                with st.expander(
                    "View Audit History"
                ):

                    try:

                        history = (
                            get_audit_history(
                                document["id"]
                            )
                        )

                        if history:

                            for item in history:

                                st.write(
                                    f"**{item.get('timestamp', '')}** — "
                                    f"{item.get('action', '')} — "
                                    f"{item.get('previous_status', '')} → "
                                    f"{item.get('new_status', '')}"
                                )

                                if item.get(
                                    "reason"
                                ):

                                    st.caption(
                                        item[
                                            "reason"
                                        ]
                                    )

                                st.divider()

                        else:

                            st.info(
                                "No audit history available."
                            )

                    except Exception as error:

                        st.error(
                            f"Audit history could not be loaded: {error}"
                        )

                with st.expander(
                    "View Full Metadata"
                ):

                    st.json(
                        dict(document)
                    )


# ==================================================
# AUDIT HISTORY
# ==================================================

elif page == "Audit History":

    st.header(
        "Audit History"
    )

    st.write(
        "Track workflow actions, status transitions, "
        "timestamps, and reviewer reasons."
    )

    st.divider()

    try:

        all_documents = filter_documents(
            search_term="",
            document_type="All",
            status="All",
            start_date=None,
            end_date=None,
            sort_order="Newest",
        )

    except Exception:

        all_documents = []

    if not all_documents:

        st.info(
            "No documents are available."
        )

    else:

        all_documents = [
            dict(row)
            for row in all_documents
        ]

        selected_document = st.selectbox(
            "Select Document",
            all_documents,
            format_func=lambda item: (
                f"{item['id']} — "
                f"{item['original_filename']}"
            ),
        )

        if selected_document:

            st.subheader(
                "Document"
            )

            st.write(
                f"**Filename:** "
                f"{selected_document['original_filename']}"
            )

            st.write(
                f"**Current Status:** "
                f"{selected_document['status']}"
            )

            try:

                history = (
                    get_audit_history(
                        selected_document["id"]
                    )
                )

            except Exception:

                history = []

            st.divider()

            if not history:

                st.info(
                    "No audit entries found for this document."
                )

            else:
                history = [dict(row) for row in history]

                for item in history:

                    st.write(
                        f"### {item.get('action', 'Action')}"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.write(
                            "**Previous Status:**",
                            item.get(
                                "previous_status"
                            )
                            or "None"
                        )

                    with col2:

                        st.write(
                            "**New Status:**",
                            item.get(
                                "new_status"
                            )
                            or "None"
                        )

                    with col3:

                        st.write(
                            "**Timestamp:**",
                            item.get(
                                "timestamp"
                            )
                            or "Unknown"
                        )

                    st.write(
                        "**Reason / Reviewer Note:**",
                        item.get(
                            "reason"
                        )
                        or "No reason recorded."
                    )

                    st.divider()