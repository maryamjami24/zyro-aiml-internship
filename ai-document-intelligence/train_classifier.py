import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from document_processor import extract_text_from_pdf


# Dataset location
DATASET_DIR = Path("dataset")

# Saved trained model
MODEL_FILE = "document_classifier.pkl"


def load_dataset():
    """
    Read PDF files from invoice and resume folders
    and create a text classification dataset.
    """

    documents = []

    for label in ["invoice", "resume"]:
        folder = DATASET_DIR / label

        for pdf_file in folder.glob("*.pdf"):
            text = extract_text_from_pdf(pdf_file)

            # Skip files with errors
            if text.startswith("Error processing PDF"):
                print(f"Skipping error file: {pdf_file.name}")
                continue

            # Skip very short documents
            if len(text.strip()) < 30:
                print(f"Skipping very short document: {pdf_file.name}")
                continue

            documents.append({
                "filename": pdf_file.name,
                "text": text,
                "label": label
            })

    return pd.DataFrame(documents)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate one classification model.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["invoice", "resume"]
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def main():

    print("Loading document dataset...")

    df = load_dataset()

    print(f"\nTotal documents loaded: {len(df)}")

    print("\nDocuments by class:")
    print(df["label"].value_counts())

    print("\nDataset:")
    print(
        df[["filename", "label"]].to_string(index=False)
    )

    # Check that both classes exist
    if df["label"].nunique() < 2:
        print("\nError: At least two document classes are required.")
        return

    X = df["text"]
    y = df["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining documents: {len(X_train)}")
    print(f"Testing documents : {len(X_test)}")

    # --------------------------------------------------
    # Model 1: TF-IDF + Logistic Regression
    # --------------------------------------------------

    logistic_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=3000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # --------------------------------------------------
    # Model 2: TF-IDF + Linear SVM
    # --------------------------------------------------

    svm_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=3000
            )
        ),
        (
            "classifier",
            LinearSVC()
        )
    ])

    # --------------------------------------------------
    # Model 3: TF-IDF + Naive Bayes
    # --------------------------------------------------

    nb_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=3000
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    # --------------------------------------------------
    # Evaluate all models
    # --------------------------------------------------

    results = []

    results.append(
        evaluate_model(
            "TF-IDF + Logistic Regression",
            logistic_model,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "TF-IDF + Linear SVM",
            svm_model,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "TF-IDF + Naive Bayes",
            nb_model,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # --------------------------------------------------
    # Compare models
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    # --------------------------------------------------
    # Train final Logistic Regression model
    # on the complete dataset
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    logistic_model.fit(X, y)

    # Save trained model
    joblib.dump(
        logistic_model,
        MODEL_FILE
    )

    print(f"\nFinal model saved successfully as: {MODEL_FILE}")

    print("\nDataset and model evaluation completed.")


if __name__ == "__main__":
    main()