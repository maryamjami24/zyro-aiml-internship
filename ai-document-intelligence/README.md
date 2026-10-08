# AI Document Intelligence

An improved Document Intelligence and Document Management application built with Python and Streamlit for the **Zyroo AI/ML Internship Program**.

The application allows users to upload documents, extract and clean text, identify document types, extract important information, validate extracted fields, manage workflow states, review documents, maintain an audit history, and store processed documents in an organized SQLite repository.

## Features

### Document Processing

* Upload PDF, JPG, JPEG, and PNG documents
* Extract text from normal PDF files using PyMuPDF
* Detect scanned PDFs and use Tesseract OCR when normal PDF text extraction is insufficient
* Extract text from images using Tesseract OCR
* Apply image preprocessing before OCR:

  * Grayscale conversion
  * Image resizing
  * Median filtering
  * Thresholding
* Clean and normalize extracted text

### Document Classification

* Identify documents as:

  * Invoice
  * Resume
  * Other
* Use a rule-based classification baseline
* Use a machine learning classifier with TF-IDF and Logistic Regression
* Compare:

  * Logistic Regression
  * Linear SVM
  * Multinomial Naive Bayes
* Display ML confidence where available

### Information Extraction

#### Invoice

* Invoice Number
* Date
* Company Name
* Total Amount

#### Resume

* Name
* Email
* Phone
* Skills

If information is missing, the application displays **Not Found** instead of failing.

---

## Week 4 Document Management

The Week 4 version extended the document processor into a document management system.

### Structured File Storage

Uploaded documents are stored according to their document type.

```text
storage/
├── invoices/
├── resumes/
└── other/
```

* Generate safe stored filenames
* Preserve the original filename in the database
* Store the final file path with each document
* Store document metadata in SQLite

### SQLite Document Repository

The repository stores information such as:

* ID
* Original filename
* Stored filename
* Document type
* Upload date
* Company
* Invoice number
* Total amount
* File path
* Text preview
* File hash
* Processing status

### Duplicate Detection

Every uploaded file is processed using a SHA-256 hash.

Before saving a new document, the application checks whether the same hash already exists.

If the file already exists:

* No new database record is created
* No new file is stored
* Existing document information is displayed
* The user is informed that the document is a duplicate

### Search and Organization

The repository supports searching across multiple document fields, including:

* Filename
* Company
* Invoice number
* Document type
* Stored text preview

### Filters and Sorting

The repository supports:

* Filter by document type
* Filter by processing status
* Filter by upload date
* Sort by newest
* Sort by oldest
* Clear filters

### Document Detail View

Saved documents can be viewed from the repository.

The detail view displays information such as:

* Original filename
* Stored filename
* Document type
* Processing status
* Company
* Invoice number
* Total amount
* File path
* Upload date
* Text preview

---

## Week 5 Advanced Document Workflow & Automation

Week 5 extends the document management system with workflow automation, validation, human review, auditability, batch processing, and workflow monitoring.

### Workflow State Management

Documents are managed using defined workflow states:

```text
New
Processing
Needs Review
Approved
Rejected
Completed
```

The current workflow state is stored in SQLite.

Workflow transitions are controlled by the workflow logic so that invalid status changes can be prevented.

### Advanced Validation

The application validates important extracted fields before a document can move through the workflow.

#### Invoice Validation

The following fields are checked:

* Invoice Number
* Date
* Company Name
* Total Amount

Validation includes:

* Required field checks
* Date format validation
* Numeric amount validation

#### Resume Validation

The following fields are checked:

* Name
* Email
* Skills
* Phone when available

Validation includes:

* Required field checks
* Email format validation
* Phone format validation

Validation failures identify the specific fields that failed.

Documents with missing or invalid required information are sent to **Needs Review**.

### Rule-Based Workflow Engine

Workflow decisions are handled separately from the Streamlit interface in `workflow.py`.

The workflow engine considers:

* Document type
* Validation results
* Missing or invalid fields
* Classification confidence when available

The workflow produces:

* A decision
* A clear reason
* Validation results

The workflow rules are kept separate from the UI so they can be modified without changing the main application interface.

### Confidence-Aware Review

When the classifier provides a confidence value, it is stored with the document.

The workflow uses a **70% confidence threshold**.

* Confidence below 70% → **Needs Review**
* Confidence at or above 70% → allowed to continue when validation passes
* No confidence value is invented when the classifier does not provide one

Documents can therefore be sent for human review when classification confidence is low.

### Human Review Queue

A dedicated review queue allows users to review documents requiring attention.

The review interface displays:

* Filename
* Document type
* Current status
* Review reason
* Extracted fields
* Validation results
* Classification confidence when available

Reviewers can:

* Approve a document
* Reject a document
* Add a review note
* Provide a rejection reason

A rejection reason is required when rejecting a document.

### Audit Log

Workflow actions are recorded in an audit history.

The audit log stores information such as:

* Document ID
* Action
* Previous status
* New status
* Timestamp
* Reason
* Reviewer note when applicable

This provides a history of important workflow changes.

### Batch Workflow Processing

Multiple documents can be selected and processed together using the same workflow rules.

The batch workflow reports:

* Processed documents
* Documents sent to Needs Review
* Failed documents
* Individual results

A failure for one document does not stop the processing of the remaining selected documents.

Documents that are already completed can be skipped instead of being processed again.

### Workflow Search and Filters

The Document Repository supports workflow-related search and filtering.

Users can search and filter documents using information such as:

* Filename
* Document type
* Company
* Invoice number
* Workflow status

Supported workflow statuses include:

* Needs Review
* Approved
* Rejected
* Completed

The repository also displays the latest document status and workflow information.

### Workflow Dashboard

A workflow dashboard provides an overview of the document processing system.

The dashboard displays:

* Total documents
* Processed documents
* Needs Review count
* Approved count
* Rejected count
* Completed count
* Failed count
* Documents by type
* Workflow state counts

This provides a quick view of the current workflow state of the document repository.

---

## Week 6 Final Integration, Testing & Optimization

Week 6 focused on final integration, reliability testing, workflow verification, persistence, optimization, security considerations, documentation, and deployment readiness.

### Final Integrated Workflow

The final application workflow is:

```text
Upload
   ↓
File Validation
   ↓
SHA-256 Hash / Duplicate Check
   ↓
PDF Text Extraction / OCR
   ↓
Text Cleaning
   ↓
Document Classification
   ↓
Information Extraction
   ↓
Field Validation
   ↓
Workflow Decision
   ↓
Needs Review / Approve / Reject / Complete
   ↓
Audit History
   ↓
Search / Filter / Dashboard
```

### Final Testing

The application was tested with:

* Valid invoice and resume documents
* Documents with missing fields
* Invalid email and phone values
* Low-confidence classification
* Duplicate documents
* OCR image documents
* Corrupted/unreadable PDF files
* Approve workflow
* Reject workflow with reviewer reason
* Audit history
* Application restart and database persistence
* Repository search and workflow filtering
* Mixed-success batch processing

### Week 6 Test Results

The final tests confirmed:

* Low-confidence documents are routed to **Needs Review**
* Invalid extracted fields are detected during validation
* Duplicate documents are detected using SHA-256 hashing
* OCR successfully processes image documents
* Corrupted/unreadable documents show a user-friendly error instead of crashing
* Approved and rejected documents maintain their workflow status
* Reviewer actions are recorded in the audit history
* Saved documents and workflow states remain available after application restart
* Repository search and workflow filters return the expected documents
* Batch processing handles multiple documents without stopping the complete batch
* Already completed documents can be skipped during batch processing
* Python syntax validation completed successfully for the main application modules

### Mixed-Success Batch Testing

A mixed batch containing documents with different processing outcomes was tested.

Example results:

```text
LAB MANUAL 1 → Needs Review
Reason: Classification confidence 56.8% is below the 70.0% threshold.

missing_fields_test_resume.pdf → Needs Review
Reason: Validation failed because Email and Phone had invalid formats.

DLD LAB WORK → Skipped
Reason: Document was already Completed.
```

The batch completed without any failed processing or application crash.

### Persistence Testing

The Streamlit application was stopped and restarted to verify database persistence.

Previously stored documents, workflow states, approval/rejection records, and dashboard statistics remained available after restart.

### Code Quality Check

The following project modules were successfully checked using Python compilation:

```text
app.py
database.py
document_processor.py
ocr_processor.py
storage_manager.py
validator.py
workflow.py
audit.py
```

No Python syntax errors were reported.

---

## Complete Processing Workflow

The document processing and management workflow is:

```text
Upload
   ↓
Validate File
   ↓
Hash / Duplicate Check
   ↓
Read / OCR
   ↓
Clean Text
   ↓
Classify
   ↓
Extract Fields
   ↓
Validate Extracted Data
   ↓
Apply Workflow Rules
   ↓
Human Review if Required
   ↓
Store File
   ↓
Store Metadata
   ↓
Audit Workflow Actions
   ↓
Search / Filter
   ↓
View / Monitor
```

---

## Week 3 Improvements

The original Week 2 MVP was improved in Week 3 to handle more realistic document differences.

### Improved Dataset

A small balanced dataset was created for document classification.

The dataset contains:

* 4 invoice documents
* 4 resume documents

### Text Cleaning and Normalization

Extracted text is cleaned before classification and information extraction.

The cleaning process handles:

* Extra spaces
* Repeated blank lines
* Unnecessary whitespace
* Empty or very short extracted text

### Improved OCR

OCR preprocessing was added to improve text recognition.

The preprocessing pipeline includes:

1. Convert image to grayscale
2. Resize the image
3. Apply median filtering
4. Apply thresholding
5. Send the processed image to Tesseract OCR

Scanned PDFs are rendered as images and processed using OCR when normal PDF text extraction is insufficient.

### Machine Learning Classification

A TF-IDF based machine learning pipeline was added.

Three models were trained and compared:

* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes

A rule-based keyword classifier is also kept as a baseline.

### Evaluation Results

The dataset contained 8 documents:

* Invoice: 4
* Resume: 4

A stratified train/test split was used:

* Training documents: 6
* Test documents: 2

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression     |     1.00 |      1.00 |   1.00 |     1.00 |
| Linear SVM              |     1.00 |      1.00 |   1.00 |     1.00 |
| Multinomial Naive Bayes |     1.00 |      1.00 |   1.00 |     1.00 |

These results represent performance on the small test split used for this internship task. A larger and more diverse dataset would be required for stronger real-world evaluation.

### Missing Field Handling

The application displays **Not Found** when an important field cannot be detected.

Example:

```text
Invoice Number: Not Found
Date: 15/09/2026
Company Name: ACME SERVICES COMPANY
Total Amount: 1,200.00$
```

### Scanned Document Testing

A scanned invoice PDF was tested successfully using OCR.

Example extracted information:

```text
Invoice Number: 7845/2026
Date: 16/09/2026
Company Name: SCAN TEST COMPANY
Total Amount: 1,000.00$
```

---

## Testing

The application was tested using different document types and workflow scenarios.

### Document Testing

Testing included:

* Normal invoice documents
* Resume documents
* Other document types
* Scanned documents
* Documents with missing fields
* Duplicate documents
* Invalid extracted fields
* Low classification confidence
* Corrupted/unreadable documents

### Workflow Testing

The workflow was tested for:

* Normal invoice processing
* Resume processing
* Missing invoice fields
* Invalid date values
* Invalid email values
* Invalid amount values
* Low classification confidence
* Needs Review workflow
* Human approval
* Human rejection
* Rejection reason handling
* Audit history
* Batch processing
* Mixed-success batch processing
* Completed document skipping
* Repository search and filtering
* Workflow dashboard metrics
* Application restart persistence

### Testing Evidence

Screenshots are included in the `screenshots` folder.

The evidence covers:

* Human Review Queue
* Audit History
* Batch Workflow
* Document Repository
* Workflow Dashboard

---

## Technologies Used

* Python
* Streamlit
* SQLite
* Pandas
* NumPy
* scikit-learn
* PyMuPDF
* Tesseract OCR
* Pytesseract
* Pillow
* Joblib
* Regular Expressions (Regex)
* hashlib

---

## Limitations

This project is still a small Document Intelligence and Document Management system.

* The training dataset is small.
* Classification results may change with larger and more diverse documents.
* OCR accuracy depends on image quality and document design.
* Stylized fonts may produce OCR errors.
* Complex document layouts may not be handled perfectly.
* Field extraction uses regex and rule-based patterns.
* Some fields may be displayed as Not Found.
* ML confidence is model-based and should not be treated as guaranteed correctness.
* Workflow rules are rule-based and intended for this internship project.
* The SQLite repository is intended for small-scale document management.

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/maryamjami24/zyro-aiml-internship.git
```

### 2. Move into the Project Directory

```bash
cd zyro-aiml-internship/ai-document-intelligence
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Classifier if Needed

```bash
python train_classifier.py
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## SQLite Database

The application creates and uses:

```text
documents.db
```

The database stores document metadata, workflow states, and audit-related information.

The required storage folders are also created for:

```text
storage/invoices
storage/resumes
storage/other
```

The database and storage system allow saved documents to remain available after restarting the Streamlit application.

---

## Supported File Types

The application supports:

* PDF
* JPG
* JPEG
* PNG

Large or unsupported files are rejected with a clear user-facing message.

---

## Success Criteria

The final project includes:

* Organized file storage
* SQLite document repository
* SHA-256 duplicate detection
* Multi-field search
* Filters and sorting
* Document detail view
* OCR and text extraction
* Machine learning classification
* Information extraction
* Advanced field validation
* Rule-based workflow engine
* Confidence-aware review
* Human review queue
* Approve and reject actions
* Audit logging
* Batch workflow processing
* Mixed-success batch handling
* Workflow search and filtering
* Workflow metrics dashboard
* Database persistence after restart
* Corrupted document handling
* Reliability and workflow testing
* Updated project documentation
* Deployment-ready project structure

---

## Author

**Maryam Jamil**

BS Computer Science Student

Zyroo AI/ML Internship Program
