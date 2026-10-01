import re

from datetime import datetime


def validate_email(email):
    """Validate an email address."""
    if not email:
        return False

    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, str(email).strip()))


def validate_phone(phone):
    """Validate a phone number when one is available."""
    if not phone:
        return True

    digits = re.sub(r"\D", "", str(phone))
    return 7 <= len(digits) <= 15


def validate_date(date_value):
    """Validate common numeric and written date formats."""
    if not date_value:
        return False

    value = str(date_value).strip()

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    for date_format in formats:
        try:
            datetime.strptime(value, date_format)
            return True
        except ValueError:
            continue

    return False


def validate_amount(amount):
    """Validate a numeric monetary amount."""
    if amount is None or str(amount).strip() == "":
        return False

    value = str(amount).strip()
    cleaned = re.sub(r"[$€£₹,\s]", "", value)

    # Allow optional decimal part.
    return bool(re.fullmatch(r"-?\d+(\.\d{1,2})?", cleaned))


def validate_invoice(data):
    """
    Validate required invoice information.

    Required:
    - Invoice Number
    - Date
    - Company Name
    - Total Amount
    """
    errors = []

    invoice_number = data.get("invoice_number")
    invoice_date = data.get("invoice_date")
    company = data.get("company")
    total_amount = data.get("total_amount")

    if not invoice_number:
        errors.append("Invoice Number is missing.")

    if not invoice_date:
        errors.append("Date is missing.")
    elif not validate_date(invoice_date):
        errors.append("Date has an invalid format.")

    if not company:
        errors.append("Company Name is missing.")

    if not total_amount:
        errors.append("Total Amount is missing.")
    elif not validate_amount(total_amount):
        errors.append("Total Amount has an invalid numeric format.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_resume(data):
    """
    Validate required resume information.

    Required:
    - Name
    - Email
    - Skills
    """
    errors = []

    name = data.get("person_name") or data.get("name")
    email = data.get("email")
    skills = data.get("skills")

    if not name:
        errors.append("Name is missing.")

    if not email:
        errors.append("Email is missing.")
    elif not validate_email(email):
        errors.append("Email has an invalid format.")

    if not skills:
        errors.append("Skills are missing.")

    phone = data.get("phone")

    if phone and not validate_phone(phone):
        errors.append("Phone has an invalid format.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_document(document_type, data):
    """
    Validate a document according to its document type.

    Returns:
        {
            "valid": bool,
            "errors": list,
            "status": "Passed" or "Failed"
        }
    """
    document_type = str(document_type or "").strip().title()

    if document_type == "Invoice":
        result = validate_invoice(data)

    elif document_type == "Resume":
        result = validate_resume(data)

    else:
        result = {
            "valid": True,
            "errors": [],
        }

    result["status"] = "Passed" if result["valid"] else "Failed"

    return result