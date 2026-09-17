import re
from pathlib import Path

import joblib
import streamlit as st

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

        return {
            "success": True,
            "file_name": file_name,
            "reading_method": reading_method,
            "cleaned_text": cleaned_text,
            "ml_type": ml_type,
            "ml_confidence": ml_confidence,
            "rule_type": rule_type,
            "invoice_fields": invoice_fields,
            "resume_fields": resume_fields
        }

    except Exception as e:

        return {
            "success": False,
            "error": f"Error processing document: {e}"
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

st.title("📄 AI Document Intelligence")

st.write(
    "Upload an invoice, resume, or document to extract "
    "text, identify its type, and find important information."
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    result = process_document(
        uploaded_file
    )

    if not result["success"]:

        st.error(
            result["error"]
        )

    else:

        # ====================================================
        # DOCUMENT ANALYSIS
        # ====================================================

        st.header("📊 Document Analysis")

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

        # ====================================================
        # INVOICE INFORMATION
        # ====================================================

        if result["invoice_fields"] is not None:

            st.header(
                "🧾 Invoice Information"
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

        # ====================================================
        # RESUME INFORMATION
        # ====================================================

        if result["resume_fields"] is not None:

            st.header(
                "👤 Resume Information"
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

        # ====================================================
        # EXTRACTED TEXT
        # ====================================================

        st.header(
            "📝 Extracted & Cleaned Text"
        )

        st.text_area(
            "Document Text",
            result["cleaned_text"],
            height=400
        )

        # ====================================================
        # PROCESSING INFORMATION
        # ====================================================

        st.header(
            "⚙️ Processing Information"
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
            "**Cleaned text length:**",
            f"{len(result['cleaned_text'])} characters"
        )