from validator import validate_document


# Confidence threshold for automatic processing.
# If confidence is below this value, the document needs human review.
CONFIDENCE_THRESHOLD = 70.0


def apply_workflow_rules(document_type, extracted_data, confidence=None):
    """
    Apply validation and confidence rules to a document.

    Returns:
        {
            "decision": "Approved" / "Needs Review",
            "reason": "...",
            "validation": {...}
        }
    """

    # Step 1: Validate extracted fields
    validation = validate_document(document_type, extracted_data)

    # Step 2: If required fields are missing or invalid,
    # send the document to human review.
    if not validation["valid"]:
        reason = "Validation failed: " + " ".join(validation["errors"])

        return {
            "decision": "Needs Review",
            "reason": reason,
            "validation": validation,
        }

    # Step 3: Use confidence only when a real confidence
    # value is available from the classifier.
    if confidence is not None:
        try:
            confidence_value = float(confidence)

            if confidence_value < CONFIDENCE_THRESHOLD:
                return {
                    "decision": "Needs Review",
                    "reason": (
                        f"Classification confidence {confidence_value:.1f}% "
                        f"is below the {CONFIDENCE_THRESHOLD:.1f}% threshold."
                    ),
                    "validation": validation,
                }

        except (TypeError, ValueError):
            return {
                "decision": "Needs Review",
                "reason": "Invalid classification confidence value.",
                "validation": validation,
            }

    # Step 4: Everything passed.
    return {
        "decision": "Approved",
        "reason": "Validation passed and workflow rules were satisfied.",
        "validation": validation,
    }