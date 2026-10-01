import database


def log_action(document_id, action, previous_status, new_status, reason=""):
    """
    Record a workflow action in the audit log.
    """

    return database.add_audit_log(
        document_id=document_id,
        action=action,
        previous_status=previous_status,
        new_status=new_status,
        reason=reason,
    )


def get_history(document_id):
    """
    Return the complete audit history for a document.
    """

    return database.get_audit_history(document_id)