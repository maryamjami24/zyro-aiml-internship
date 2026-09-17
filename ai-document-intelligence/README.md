# AI Document Intelligence

An improved Document Intelligence application built with Python and Streamlit for the **Zyroo AI/ML Internship Program**.

The application allows users to upload documents, extract and clean text, identify document types, and extract important information from invoices and resumes.

## Features

- Upload PDF, JPG, JPEG, and PNG documents
- Extract text from normal PDF files using PyMuPDF
- Detect scanned PDFs and use Tesseract OCR when normal PDF text extraction is insufficient
- Extract text from images using Tesseract OCR
- Apply image preprocessing before OCR:
  - Grayscale conversion
  - Image resizing
  - Median filtering
  - Thresholding
- Clean and normalize extracted text
- Identify documents as:
  - Invoice
  - Resume
  - Other
- Use a rule-based classification baseline
- Use a machine learning classifier with TF-IDF and Logistic Regression
- Compare multiple simple ML models:
  - Logistic Regression
  - Linear SVM
  - Multinomial Naive Bayes
- Display ML confidence where available
- Extract invoice information:
  - Invoice Number
  - Date
  - Company Name
  - Total Amount
- Extract resume information:
  - Name
  - Email
  - Phone
  - Skills
- Display **Not Found** when information is missing
- Display the reading method used:
  - PDF text extraction
  - OCR (scanned PDF)
  - Image OCR
- Display cleaned text and its length
- Streamlit-based user interface

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- PyMuPDF
- Tesseract OCR
- Pytesseract
- Pillow
- Joblib
- Regular Expressions (Regex)

## Project Structure

```text
ai-document-intelligence/

│
├── app.py
├── document_processor.py
├── ocr_processor.py
├── train_classifier.py
├── document_classifier.pkl
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
└── screenshots/
Week 2 to Week 3 Improvements

The original Week 2 MVP was improved in Week 3 to handle more realistic document differences.

1. Improved Dataset

A small balanced dataset was created for document classification.

The dataset contains:

4 invoice documents
4 resume documents

Unclear or personal documents were avoided in the training dataset so that the classifier could focus on the two document classes.

2. Text Cleaning and Normalization

Extracted text is cleaned before classification and information extraction.

The cleaning process handles:

Extra spaces
Repeated blank lines
Unnecessary whitespace
Empty or very short extracted text

The cleaned text is then used for classification and extraction.

3. Improved OCR

OCR preprocessing was added to improve text recognition.

The image preprocessing pipeline includes:

Convert image to grayscale
Resize the image
Apply median filtering
Apply thresholding
Send the processed image to Tesseract OCR

Scanned PDFs are also handled by rendering their pages as images and applying OCR when normal PDF text extraction is insufficient.

4. OCR Testing

OCR was tested using an image containing text from DevelopersHub Corporation.

The original and preprocessed OCR results were compared.

Common OCR errors included:

Stylized text was sometimes recognized incorrectly.
Some text such as a URL was partially detected.
Normal body text was recognized more accurately.
Preprocessing improved recognition of some text areas but did not completely remove OCR errors.

A scanned invoice PDF was also tested successfully.

The scanned invoice was processed using:

OCR (scanned PDF)

and the following information was extracted:

Invoice Number: 7845/2026
Date: 16/09/2026
Company Name: SCAN TEST COMPANY
Total Amount: 1,000.00$
Machine Learning Classification

A simple machine learning classification pipeline was added using:

TF-IDF → Classifier

Three models were trained and compared using the same dataset:

Logistic Regression
Linear SVM
Multinomial Naive Bayes

A rule-based keyword classifier is also kept as a baseline.

Evaluation Results

The dataset contained 8 documents:

Invoice: 4
Resume: 4

A stratified train/test split was used:

Training documents: 6
Test documents: 2
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	1.00	1.00	1.00	1.00
Linear SVM	1.00	1.00	1.00	1.00
Multinomial Naive Bayes	1.00	1.00	1.00	1.00

The confusion matrix for the evaluated test split was:

[[1 0]
 [0 1]]

These results represent performance on the small test split used for this internship task. A larger and more diverse dataset would be required to make stronger conclusions about real-world classification performance.

Improved Information Extraction
Invoice Extraction

The application extracts:

Invoice Number
Date
Company Name
Total Amount

The extraction rules were improved to handle different invoice formats and common labels such as total amount, grand total, and balance due.

Resume Extraction

The application extracts:

Name
Email
Phone
Skills

Different resume layouts were tested to check whether the extraction rules could handle document variations.

Missing Field Handling

The application does not fail when a field is missing.

Instead, the field is displayed as:

Not Found

A resume with missing email and phone information was tested.

Example:

Name: JOHN SMITH
Email: Not Found
Phone: Not Found
Skills: Figma, Adobe XD, UI Design, Communication, Wireframing

A test invoice with a missing invoice number was also processed successfully.

Example:

Invoice Number: Not Found
Date: 15/09/2026
Company Name: ACME SERVICES COMPANY
Total Amount: 1,200.00$

This demonstrates that missing information is handled without breaking the application.

Document Processing Workflow

The improved workflow is:

Upload → Read Text/OCR → Clean Text → Identify Type → Extract Fields → Check Missing Fields → Show Result

1. Upload Document

The user uploads a PDF, JPG, JPEG, or PNG document through the Streamlit interface.

2. Read Text
Normal PDFs are processed using PyMuPDF.
Scanned PDFs are detected when normal text extraction is insufficient and processed using OCR.
Images are processed using Tesseract OCR.
3. Clean Text

The extracted text is normalized by removing unnecessary whitespace and repeated blank lines.

4. Identify Document Type

The application uses both:

Rule-based classification
TF-IDF machine learning classification

The ML classifier predicts the document type and provides confidence information where available.

5. Extract Information

The application extracts relevant fields depending on the document type.

6. Handle Missing Fields

Fields that cannot be detected are displayed as Not Found.

7. Show Results

The application displays:

Document type
ML confidence
Rule-based type
Extracted fields
Reading method
Cleaned text
Cleaned text length
Testing

The improved application was tested with multiple document types and formats.

Invoice Tests

Four different invoice templates were tested:

Construction Invoice
Contractor Invoice
Proforma Invoice
Rental Invoice

The application successfully identified these documents as invoices and extracted invoice-related fields.

Resume Tests

Multiple resume templates were tested to check classification and extraction across different layouts.

Missing Field Tests

Documents with missing fields were tested to confirm that the application displays Not Found instead of failing.

Scanned Document Test

A scanned invoice PDF containing image-based text was tested.

The application correctly detected that normal PDF text extraction was insufficient and used:

OCR (scanned PDF)

The scanned invoice information was successfully extracted.

Limitations

This project is still a small Document Intelligence system.

The training dataset is small.
Classification results may change with larger and more diverse documents.
OCR accuracy depends on image quality and document design.
Stylized fonts may produce OCR errors.
Complex document layouts may not be handled perfectly.
Field extraction uses regex and rule-based patterns.
Some fields may be displayed as Not Found.
ML confidence is model-based and should not be treated as guaranteed correctness.
How to Run Locally

Clone the repository:

git clone https://github.com/maryamjami24/zyro-aiml-internship.git

Move into the project directory:

cd zyro-aiml-internship/ai-document-intelligence

Install the required dependencies:

pip install -r requirements.txt

Train the classifier if needed:

python train_classifier.py

Run the Streamlit application:

streamlit run app.py

The application will open in the browser.

Success Criteria

The Week 3 version improves the original MVP by adding:

A balanced document dataset
Text cleaning and normalization
Improved OCR preprocessing
Scanned PDF OCR support
TF-IDF based machine learning classification
Comparison of multiple simple ML models
Accuracy, precision, recall, F1 score, and confusion matrix evaluation
Improved invoice and resume extraction
Missing-field handling
ML confidence information
Testing with multiple invoices, resumes, missing fields, and scanned documents
Author

Maryam Jamil

BS Computer Science Student

Zyroo AI/ML Internship Program