import streamlit as st
import re
from document_processor import extract_text_from_pdf
from ocr_processor import extract_text_from_image


st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Document Classification
# -----------------------------
def classify_document(text):
    text_lower = text.lower()

    invoice_keywords = [
        "invoice",
        "invoice number",
        "invoice no",
        "total",
        "amount due"
    ]

    resume_keywords = [
        "resume",
        "curriculum vitae",
        "skills",
        "education",
        "experience",
        "work experience"
    ]

    invoice_score = sum(
        1 for keyword in invoice_keywords
        if keyword in text_lower
    )

    resume_score = sum(
        1 for keyword in resume_keywords
        if keyword in text_lower
    )

    if invoice_score >= 2:
        return "Invoice"

    if resume_score >= 2:
        return "Resume"

    return "Other"


# -----------------------------
# Invoice Field Extraction
# -----------------------------
def extract_invoice_fields(text):
    fields = {
        "Invoice Number": "Not found",
        "Date": "Not found",
        "Company Name": "Not found",
        "Total Amount": "Not found"
    }

    # Invoice Number
    invoice_patterns = [
        r"(?:invoice\s*(?:number|no\.?|#))\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        r"(?:inv\s*(?:number|no\.?|#))\s*[:\-]?\s*([A-Za-z0-9\-\/]+)"
    ]

    for pattern in invoice_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields["Invoice Number"] = match.group(1)
            break

    # Date
    date_patterns = [
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        r"\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields["Date"] = match.group(0)
            break

    # Total Amount
    total_patterns = [
        r"(?:grand\s+total|total\s+amount|amount\s+due|total)\s*[:\-]?\s*(?:Rs\.?|PKR|\$|USD)?\s*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"(?:Rs\.?|PKR|\$|USD)\s*([0-9,]+(?:\.[0-9]{1,2})?)"
    ]

    for pattern in total_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            fields["Total Amount"] = matches[-1]
            break

    # Company Name
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:8]:
        line_lower = line.lower()

        if (
            "invoice" not in line_lower
            and "date" not in line_lower
            and "total" not in line_lower
            and not re.search(r"\d", line)
        ):
            fields["Company Name"] = line
            break

    return fields


# -----------------------------
# Resume Field Extraction
# -----------------------------
def extract_resume_fields(text):
    fields = {
        "Name": "Not found",
        "Email": "Not found",
        "Phone": "Not found",
        "Skills": "Not found"
    }

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Name
    if lines:
        first_line = lines[0]

        if (
            len(first_line.split()) <= 5
            and not re.search(r"@|\d", first_line)
            and first_line.lower() not in ["resume", "curriculum vitae"]
        ):
            fields["Name"] = first_line

    # Email
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if email_match:
        fields["Email"] = email_match.group(0)

    # Phone
    phone_patterns = [
        r"\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\+?\d{1,4}[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    ]

    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)

        if phone_match:
            fields["Phone"] = phone_match.group(0)
            break

    # Skills
    common_skills = [
        "Python",
        "Java",
        "C++",
        "C#",
        "JavaScript",
        "HTML",
        "CSS",
        "SQL",
        "Machine Learning",
        "Artificial Intelligence",
        "Data Science",
        "Data Analysis",
        "Git",
        "GitHub",
        "React",
        "Node.js",
        "Django",
        "Flask",
        "Microsoft Office",
        "MS Word",
        "MS PowerPoint"
    ]

    found_skills = []

    text_lower = text.lower()

    for skill in common_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)

    if found_skills:
        fields["Skills"] = ", ".join(found_skills)

    return fields


# -----------------------------
# Main UI
# -----------------------------
st.title("📄 AI Document Intelligence")

st.write(
    "Upload a PDF or image to extract text, identify the document type, "
    "and find basic information automatically."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    st.success(f"File uploaded: {uploaded_file.name}")

    file_type = uploaded_file.type

    st.write(f"**File Type:** {file_type}")

    if st.button("🚀 Process Document", type="primary"):

        text = ""

        # -----------------------------
        # PDF Processing
        # -----------------------------
        if file_type == "application/pdf":

            with open("temp_document.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Reading PDF..."):
                text = extract_text_from_pdf("temp_document.pdf")

        # -----------------------------
        # Image OCR Processing
        # -----------------------------
        else:

            with open("temp_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Running OCR..."):
                text = extract_text_from_image("temp_image.png")

        # -----------------------------
        # Display Results
        # -----------------------------
        if text and text.strip():

            document_type = classify_document(text)

            st.divider()

            st.subheader("📊 Document Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Document Type", document_type)

            with col2:
                st.metric("File Name", uploaded_file.name)

            # Invoice
            if document_type == "Invoice":

                st.subheader("🧾 Invoice Information")

                fields = extract_invoice_fields(text)

                for field, value in fields.items():
                    st.write(f"**{field}:** {value}")

            # Resume
            elif document_type == "Resume":

                st.subheader("👤 Resume Information")

                fields = extract_resume_fields(text)

                for field, value in fields.items():
                    st.write(f"**{field}:** {value}")

            # Other
            else:

                st.info(
                    "This document could not be confidently identified "
                    "as an Invoice or Resume."
                )

            # Extracted Text
            st.divider()

            st.subheader("📑 Extracted Text")

            st.text_area(
                "Document Content",
                text,
                height=400
            )

        else:

            st.error(
                "No text could be extracted from this document. "
                "Please try another file."
            )