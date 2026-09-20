import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "documents.db"


def get_connection():
    """Create and return a SQLite database connection."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the documents table if it does not already exist."""
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
            status TEXT NOT NULL DEFAULT 'Processed'
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
    status="Processed",
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
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ORDER BY upload_date DESC
            """,
            (
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
    """Search, filter, and sort documents directly in SQLite."""

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
    """Update the processing status of a document."""
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
):
    """Update editable document metadata."""

    connection = get_connection()

    fields = []
    values = []

    if company is not None:
        fields.append("company = ?")
        values.append(company)

    if invoice_number is not None:
        fields.append("invoice_number = ?")
        values.append(invoice_number)

    if total_amount is not None:
        fields.append("total_amount = ?")
        values.append(total_amount)

    if document_type is not None:
        fields.append("document_type = ?")
        values.append(document_type)

    if status is not None:
        fields.append("status = ?")
        values.append(status)

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


def delete_document(document_id):
    """Delete a document record from the database."""
    connection = get_connection()

    connection.execute(
        """
        DELETE FROM documents
        WHERE id = ?
        """,
        (document_id,),
    )

    connection.commit()
    connection.close()