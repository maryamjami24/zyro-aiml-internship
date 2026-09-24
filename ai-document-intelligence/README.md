# AI Document Intelligence

An improved Document Intelligence and Document Management application built with Python and Streamlit for the **Zyroo AI/ML Internship Program**.

The application allows users to upload documents, extract and clean text, identify document types, extract important information, and store processed documents in an organized repository with SQLite metadata.

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

### Week 4 Document Management

The Week 4 version extends the document processor into a small document management system.

* Store files in separate folders:

  * `invoices`
  * `resumes`
  * `other`
* Generate safe stored filenames
* Preserve the original filename in the database
* Store the final file path with each document
* Store document metadata in SQLite
* Calculate SHA-256 file hashes
* Detect duplicate uploads before creating a new record
* Search documents using multiple fields
* Filter documents by:

  * Document type
  * Processing status
  * Upload date
* Sort documents by newest or oldest
* Clear repository filters
* View saved document details
* Display extracted metadata and text preview
* Display processing status:

  * Processed
  * Needs Review
  * Failed
* Handle missing important fields using **Needs Review**
* Reject unsupported file types
* Limit large uploads
* Handle processing errors without exposing raw exception details to normal users

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

## Project Structure

```text
ai-document-intelligence/

│
├── app.py
├── document_processor.py
├── ocr_processor.py
├── database.py
├── storage_manager.py
├── train_classifier.py
├── document_classifier.pkl
├── documents.db
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── invoice/
│   │   ├── Construction-Invoice-Template-TemplateLab.com_.pdf
│   │   ├── Contractor-Invoice-Template-TemplateLab.com_.pdf
│   │   ├── Proforma-Invoice-Template-TemplateLab.com_.pdf
│   │   └── Rental-Invoice-Template-TemplateLab.com_.pdf
│   │
│   └── resume/
│       ├── New-York-Resume-Template-Creative.pdf
│       ├── Personal-trainer-resume-example-3.pdf
│       ├── Stockholm-Resume-Template-Simple.pdf
│       └── Ux-designer-resume-example-5.pdf
│
├── storage/
│   ├── invoices/
│   ├── resumes/
│   └── other/
│
└── screenshots/
    ├── 07_week4_document_repository.png
    ├── 08_week4_duplicate_detection.png
    └── 09_week4_needs_review.png
```

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

A test invoice with a missing invoice number was processed successfully.

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

## Week 4 Document Management

Week 4 extends the previous document processing system into a document management layer.

### Structured File Storage

Uploaded documents are stored according to their document type.

```text
storage/
├── invoices/
├── resumes/
└── other/
```

Safe filenames are generated for stored files instead of relying only on the original filename.

The original filename is preserved in the SQLite database.

### SQLite Document Repository

A SQLite database is used to store document metadata.

The repository keeps information such as:

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

The database logic is kept separate from the Streamlit interface.

### Duplicate Detection

Every uploaded file is processed using a SHA-256 hash.

Before saving a new document, the application checks whether the same hash already exists.

If the file already exists:

* No new database record is created
* No new file is stored
* The existing document information is displayed
* The user is informed that the document is a duplicate

### Search and Organization

The repository provides search functionality across multiple document fields, including:

* Filename
* Company
* Invoice number
* Document type
* Stored text preview

SQLite queries are used to search the stored document records.

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

### Processing Status

Each document can have a processing status.

The supported statuses are:

* **Processed**
* **Needs Review**
* **Failed**

Documents with missing important information can be marked as **Needs Review**.

Processing errors can result in a **Failed** status without exposing raw exception details to normal users.

## Week 4 Processing Workflow

The complete Week 4 workflow is:

```text
Upload
   ↓
Validate
   ↓
Hash
   ↓
Read / OCR
   ↓
Clean
   ↓
Classify
   ↓
Extract
   ↓
Store File
   ↓
Store Metadata
   ↓
Search / Filter
   ↓
View
```

## Testing

The Week 4 document management system was tested using different document types and processing scenarios.

Testing included:

* Invoice documents
* Resume documents
* Other document types
* Duplicate documents
* Scanned documents
* Documents with missing fields
* Repository storage
* Document search
* Filtering and sorting
* Processing status handling

### Testing Evidence

Week 4 screenshots are included in the `screenshots` folder.

#### Document Repository

`07_week4_document_repository.png`

Shows the saved documents in the document repository.

#### Duplicate Detection

`08_week4_duplicate_detection.png`

Shows that an already stored document is detected as a duplicate and no new record is created.

#### Needs Review

`09_week4_needs_review.png`

Shows a document with missing important information being marked as **Needs Review**.

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
* The SQLite repository is intended for this internship project and small-scale document management.

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

## SQLite Database

The application creates and uses:

```text
documents.db
```

The database stores document metadata and repository information.

The required storage folders are also created for:

```text
storage/invoices
storage/resumes
storage/other
```

The database and storage system allow saved documents to remain available after restarting the Streamlit application.

## Supported File Types

The application supports:

* PDF
* JPG
* JPEG
* PNG

Large or unsupported files are rejected with a clear user-facing message.

## Success Criteria

The Week 4 version extends the previous MVP by adding:

* Organized file storage
* SQLite document repository
* Document metadata storage
* SHA-256 duplicate detection
* Multi-field search
* Filters and sorting
* Document detail view
* Processing status
* Missing-field review handling
* Safer error handling
* Repository testing
* Testing screenshots
* Updated project documentation

## Author

**Maryam Jamil**

BS Computer Science Student

Zyroo AI/ML Internship Program
