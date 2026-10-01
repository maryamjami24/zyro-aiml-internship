import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "documents.db"


WORKFLOW_STATES = [
    "New",
    "Processing",
    "Needs Review",
    "Approved",
    "Rejected",
    "Completed",
]


VALID_TRANSITIONS = {
    "New": ["Processing"],
    "Processing": ["Needs Review", "Approved", "Completed"],
    "Needs Review": ["Approved", "Rejected"],
    "Approved": ["Completed"],
    "Rejected": ["Completed"],
    "Completed": [],
}


def get_connection():
    """Create and return a SQLite database connection."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _add_column_if_missing(connection, column_name, column_definition):
    """Add a column to an existing table if it does not exist."""
    columns = connection.execute(
        "PRAGMA table_info(documents)"
    ).fetchall()

    existing_columns = {column["name"] for column in columns}

    if column_name not in existing_columns:
        connection.execute(
            f"ALTER TABLE documents ADD COLUMN {column_name} {column_definition}"
        )


def initialize_database():
    """Create and upgrade the database for the Week 5 workflow."""

    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            document_type TEXT,
            upload_date TEXT NOT NULL,
            company TEXT,
            invoice_number TEXT,
            total_amount TEXT,
            file_path TEXT NOT NULL,
            text_preview TEXT,
            file_hash TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'New'
        )
        """
    )

    # Week 5 workflow fields
    _add_column_if_missing(
        connection,
        "predicted_type",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "confidence",
        "REAL"
    )

    _add_column_if_missing(
        connection,
        "review_reason",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "validation_status",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "validation_errors",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "processed_at",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "completed_at",
        "TEXT"
    )

    # Resume fields
    _add_column_if_missing(
        connection,
        "person_name",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "email",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "phone",
        "TEXT"
    )

    _add_column_if_missing(
        connection,
        "skills",
        "TEXT"
    )

    # Existing Week 4 "Processed" records are treated as completed
    # because they were already processed before Week 5.
    connection.execute(
        """
        UPDATE documents
        SET status = 'Completed'
        WHERE status = 'Processed'
        """
    )

    # Audit table
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            previous_status TEXT,
            new_status TEXT,
            timestamp TEXT NOT NULL,
            reason TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """
    )

    connection.commit()
    connection.close()


def add_document(
    original_filename,
    stored_filename,
    document_type,
    company,
    invoice_number,
    total_amount,
    file_path,
    text_preview,
    file_hash,
    status="New",
    person_name=None,
    email=None,
    phone=None,
    skills=None,
    predicted_type=None,
    confidence=None,
):
    """Add a new document to the database."""

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                original_filename,
                stored_filename,
                document_type,
                upload_date,
                company,
                invoice_number,
                total_amount,
                file_path,
                text_preview,
                file_hash,
                status,
                person_name,
                email,
                phone,
                skills,
                predicted_type,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_filename,
                stored_filename,
                document_type,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                company,
                invoice_number,
                total_amount,
                file_path,
                text_preview,
                file_hash,
                status,
                person_name,
                email,
                phone,
                skills,
                predicted_type,
                confidence,
            ),
        )

        connection.commit()
        return cursor.lastrowid

    finally:
        connection.close()


def get_document_by_hash(file_hash):
    """Find a document using its SHA-256 hash."""

    connection = get_connection()

    document = connection.execute(
        """
        SELECT *
        FROM documents
        WHERE file_hash = ?
        """,
        (file_hash,),
    ).fetchone()

    connection.close()

    return document


def get_document(document_id):
    """Get one document by its ID."""

    connection = get_connection()

    document = connection.execute(
        """
        SELECT *
        FROM documents
        WHERE id = ?
        """,
        (document_id,),
    ).fetchone()

    connection.close()

    return document


def get_all_documents():
    """Return all documents, newest first."""

    connection = get_connection()

    documents = connection.execute(
        """
        SELECT *
        FROM documents
        ORDER BY upload_date DESC
        """
    ).fetchall()

    connection.close()

    return documents


def search_documents(search_term=""):
    """Search documents across multiple fields."""

    connection = get_connection()

    if not search_term:
        documents = connection.execute(
            """
            SELECT *
            FROM documents
            ORDER BY upload_date DESC
            """
        ).fetchall()

    else:
        pattern = f"%{search_term}%"

        documents = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE
                original_filename LIKE ?
                OR company LIKE ?
                OR invoice_number LIKE ?
                OR document_type LIKE ?
                OR text_preview LIKE ?
                OR person_name LIKE ?
                OR email LIKE ?
                OR skills LIKE ?
            ORDER BY upload_date DESC
            """,
            (
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
            ),
        ).fetchall()

    connection.close()

    return documents


def filter_documents(
    search_term="",
    document_type="All",
    status="All",
    start_date=None,
    end_date=None,
    sort_order="Newest",
):
    """Search, filter and sort documents directly in SQLite."""

    connection = get_connection()

    conditions = []
    parameters = []

    if search_term:
        pattern = f"%{search_term}%"

        conditions.append(
            """
            (
                original_filename LIKE ?
                OR company LIKE ?
                OR invoice_number LIKE ?
                OR document_type LIKE ?
                OR text_preview LIKE ?
                OR person_name LIKE ?
                OR email LIKE ?
                OR skills LIKE ?
            )
            """
        )

        parameters.extend(
            [
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
            ]
        )

    if document_type and document_type != "All":
        conditions.append("document_type = ?")
        parameters.append(document_type)

    if status and status != "All":
        conditions.append("status = ?")
        parameters.append(status)

    if start_date:
        conditions.append("date(upload_date) >= ?")
        parameters.append(start_date)

    if end_date:
        conditions.append("date(upload_date) <= ?")
        parameters.append(end_date)

    query = "SELECT * FROM documents"

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if sort_order == "Oldest":
        query += " ORDER BY upload_date ASC"
    else:
        query += " ORDER BY upload_date DESC"

    documents = connection.execute(
        query,
        parameters,
    ).fetchall()

    connection.close()

    return documents


def update_document_status(document_id, status):
    """Direct status update kept for backward compatibility."""

    connection = get_connection()

    connection.execute(
        """
        UPDATE documents
        SET status = ?
        WHERE id = ?
        """,
        (
            status,
            document_id,
        ),
    )

    connection.commit()
    connection.close()


def update_document(
    document_id,
    company=None,
    invoice_number=None,
    total_amount=None,
    document_type=None,
    status=None,
    person_name=None,
    email=None,
    phone=None,
    skills=None,
    predicted_type=None,
    confidence=None,
    review_reason=None,
    validation_status=None,
    validation_errors=None,
):
    """Update document metadata and workflow information."""

    connection = get_connection()

    fields = []
    values = []

    updates = {
        "company": company,
        "invoice_number": invoice_number,
        "total_amount": total_amount,
        "document_type": document_type,
        "status": status,
        "person_name": person_name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "predicted_type": predicted_type,
        "confidence": confidence,
        "review_reason": review_reason,
        "validation_status": validation_status,
        "validation_errors": validation_errors,
    }

    for field, value in updates.items():
        if value is not None:
            fields.append(f"{field} = ?")
            values.append(value)

    if status in ["Processing", "Needs Review", "Approved", "Completed"]:
        fields.append("processed_at = ?")
        values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if status == "Completed":
        fields.append("completed_at = ?")
        values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not fields:
        connection.close()
        return

    values.append(document_id)

    connection.execute(
        f"""
        UPDATE documents
        SET {", ".join(fields)}
        WHERE id = ?
        """,
        values,
    )

    connection.commit()
    connection.close()


def transition_document(document_id, new_status, reason=""):
    """
    Change workflow state only when the transition is valid.
    Returns (success, message).
    """

    connection = get_connection()

    document = connection.execute(
        """
        SELECT status
        FROM documents
        WHERE id = ?
        """,
        (document_id,),
    ).fetchone()

    if document is None:
        connection.close()
        return False, "Document not found."

    current_status = document["status"]

    allowed_states = VALID_TRANSITIONS.get(current_status, [])

    if new_status not in allowed_states:
        connection.close()

        return (
            False,
            f"Invalid transition: {current_status} → {new_status}"
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    connection.execute(
        """
        UPDATE documents
        SET status = ?,
            processed_at = ?
        WHERE id = ?
        """,
        (
            new_status,
            now,
            document_id,
        ),
    )

    if new_status == "Completed":
        connection.execute(
            """
            UPDATE documents
            SET completed_at = ?
            WHERE id = ?
            """,
            (
                now,
                document_id,
            ),
        )

    connection.execute(
        """
        INSERT INTO audit_log (
            document_id,
            action,
            previous_status,
            new_status,
            timestamp,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            "Status Change",
            current_status,
            new_status,
            now,
            reason,
        ),
    )

    connection.commit()
    connection.close()

    return True, f"Status changed to {new_status}."


def add_audit_log(
    document_id,
    action,
    previous_status=None,
    new_status=None,
    reason="",
):
    """Record an important workflow event."""

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO audit_log (
            document_id,
            action,
            previous_status,
            new_status,
            timestamp,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            action,
            previous_status,
            new_status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason,
        ),
    )

    connection.commit()
    connection.close()


def get_audit_history(document_id):
    """Return audit history for one document."""

    connection = get_connection()

    history = connection.execute(
        """
        SELECT *
        FROM audit_log
        WHERE document_id = ?
        ORDER BY timestamp DESC, id DESC
        """,
        (document_id,),
    ).fetchall()

    connection.close()

    return history


def get_documents_by_status(status):
    """Return documents in a particular workflow state."""

    connection = get_connection()

    documents = connection.execute(
        """
        SELECT *
        FROM documents
        WHERE status = ?
        ORDER BY upload_date DESC
        """,
        (status,),
    ).fetchall()

    connection.close()

    return documents


def get_workflow_counts():
    """Return document counts for the workflow dashboard."""

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM documents
        GROUP BY status
        """
    ).fetchall()

    by_type_rows = connection.execute(
        """
        SELECT document_type, COUNT(*) AS count
        FROM documents
        GROUP BY document_type
        """
    ).fetchall()

    connection.close()

    status_counts = {
        state: 0
        for state in WORKFLOW_STATES
    }

    for row in rows:
        status = row["status"]
        if status in status_counts:
            status_counts[status] = row["count"]

    by_type = {
        row["document_type"]: row["count"]
        for row in by_type_rows
    }

    total = sum(status_counts.values())

    processed = (
        status_counts.get("Processing", 0)
        + status_counts.get("Approved", 0)
        + status_counts.get("Rejected", 0)
        + status_counts.get("Completed", 0)
    )

    return {
        "total": total,
        "processed": processed,
        "needs_review": status_counts.get("Needs Review", 0),
        "approved": status_counts.get("Approved", 0),
        "rejected": status_counts.get("Rejected", 0),
        "completed": status_counts.get("Completed", 0),
        "failed": 0,
        "by_type": by_type,
        **{
            state: count
            for state, count in status_counts.items()
        },
    }


def delete_document(document_id):
    """Delete a document and its audit history."""

    connection = get_connection()

    connection.execute(
        """
        DELETE FROM audit_log
        WHERE document_id = ?
        """,
        (document_id,),
    )

    connection.execute(
        """
        DELETE FROM documents
        WHERE id = ?
        """,
        (document_id,),
    )

    connection.commit()
    connection.close()