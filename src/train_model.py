from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from rules import (
    is_answer_correct,
    has_distributive_error,
    has_sign_error,
    has_like_terms_error,
    has_exponent_error
)


# --------------------------------
# Project Paths
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIRECTORY / "train.csv"
TEST_PATH = DATA_DIRECTORY / "test.csv"


# --------------------------------
# Model Configuration
# --------------------------------

RANDOM_SEED = 42

CLASS_NAMES = [
    "correct",
    "distributive_error",
    "exponent_error",
    "like_terms_error",
    "sign_error"
]


# --------------------------------
# Load Dataset
# --------------------------------

def load_dataset():
    """Loads the training and testing datasets."""

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    return train_data, test_data


# --------------------------------
# Prepare Features
# --------------------------------

def prepare_features(train_data, test_data):
    """Converts problems and student answers into TF-IDF features."""

    train_text = (
            train_data["problem"].astype(str)
            + " "
            + train_data["student_answer"].astype(str)
    )

    test_text = (
            test_data["problem"].astype(str)
            + " "
            + test_data["student_answer"].astype(str)
    )

    vectorizer = TfidfVectorizer()

    train_features = vectorizer.fit_transform(
        train_text
    )

    test_features = vectorizer.transform(
        test_text
    )

    return (
        vectorizer,
        train_features,
        test_features
    )


# --------------------------------
# Train Machine Learning Model
# --------------------------------

def train_classifier(
        train_features,
        train_labels
):
    """Trains the Logistic Regression misconception classifier."""

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED
    )

    model.fit(
        train_features,
        train_labels
    )

    return model


# --------------------------------
# Rule-Only Prediction
# --------------------------------

def predict_with_rules(
        problem,
        student_answer
):
    """
    Predicts a misconception using only deterministic rules.

    Returns "unknown" when no rule matches.
    """

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    # Check whether the answer is correct
    if is_answer_correct(
            problem,
            student_answer
    ):
        return "correct"

    # Check known misconception patterns
    if has_distributive_error(
            problem,
            student_answer
    ):
        return "distributive_error"

    if has_sign_error(
            problem,
            student_answer
    ):
        return "sign_error"

    if has_like_terms_error(
            problem,
            student_answer
    ):
        return "like_terms_error"

    if has_exponent_error(
            problem,
            student_answer
    ):
        return "exponent_error"

    # No rule recognized the response
    return "unknown"


def make_rule_predictions(test_data):
    """Generates predictions using only the rule-based system."""

    predictions = []

    for _, row in test_data.iterrows():
        prediction = predict_with_rules(
            row["problem"],
            row["student_answer"]
        )

        predictions.append(
            prediction
        )

    return predictions


# --------------------------------
# Hybrid Prediction
# --------------------------------

def predict_misconception(
        problem,
        student_answer,
        ml_features,
        model
):
    """
    Predicts whether an answer is correct or represents
    a supported algebra misconception.

    The hybrid system checks correctness first, applies
    deterministic misconception rules second, and uses
    machine learning as a fallback.
    """

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    # Stage 1: Check whether the answer is correct
    if is_answer_correct(
            problem,
            student_answer
    ):
        return "correct"

    # Stage 2: Apply misconception rules
    if has_distributive_error(
            problem,
            student_answer
    ):
        return "distributive_error"

    if has_sign_error(
            problem,
            student_answer
    ):
        return "sign_error"

    if has_like_terms_error(
            problem,
            student_answer
    ):
        return "like_terms_error"

    if has_exponent_error(
            problem,
            student_answer
    ):
        return "exponent_error"

    # Stage 3: Fall back to machine learning
    prediction = model.predict(
        ml_features
    )

    return prediction[0]


def make_hybrid_predictions(
        test_data,
        test_features,
        model
):
    """Generates predictions using the hybrid AI pipeline."""

    predictions = []

    for index, row in test_data.iterrows():
        prediction = predict_misconception(
            row["problem"],
            row["student_answer"],
            test_features[index],
            model
        )

        predictions.append(
            prediction
        )

    return predictions


# --------------------------------
# Standard Model Evaluation
# --------------------------------

def evaluate_model(
        test_labels,
        predictions,
        model_name
):
    """Prints classification performance for a model."""

    accuracy = accuracy_score(
        test_labels,
        predictions
    )

    print(
        f"\n{model_name} Results"
    )

    print("=" * 50)

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            test_labels,
            predictions,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )

    return accuracy


# --------------------------------
# Rule-Only Evaluation
# --------------------------------

def evaluate_rule_system(
        test_labels,
        rule_predictions
):
    """
    Evaluates the rule-only system using accuracy,
    coverage, and accuracy when a rule matches.
    """

    total_predictions = len(
        rule_predictions
    )

    recognized_count = sum(
        prediction != "unknown"
        for prediction in rule_predictions
    )

    correct_count = sum(
        prediction == actual
        for prediction, actual in zip(
            rule_predictions,
            test_labels
        )
    )

    coverage = (
            recognized_count
            / total_predictions
    )

    accuracy = (
            correct_count
            / total_predictions
    )

    if recognized_count > 0:
        recognized_correct = sum(
            prediction == actual
            for prediction, actual in zip(
                rule_predictions,
                test_labels
            )
            if prediction != "unknown"
        )

        recognized_accuracy = (
                recognized_correct
                / recognized_count
        )

    else:
        recognized_accuracy = 0

    print(
        "\nRule-Only System Results"
    )

    print("=" * 50)

    print(
        f"\nOverall Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Rule Coverage:    "
        f"{coverage:.4f}"
    )

    print(
        f"Accuracy When a Rule Matches: "
        f"{recognized_accuracy:.4f}"
    )

    print(
        f"\nRecognized samples: "
        f"{recognized_count}/"
        f"{total_predictions}"
    )

    return (
        accuracy,
        coverage,
        recognized_accuracy
    )


# --------------------------------
# Confusion Matrix
# --------------------------------

def save_confusion_matrix(
        test_labels,
        predictions,
        model_name,
        file_name
):
    """Creates and saves a confusion matrix."""

    RESULTS_DIRECTORY.mkdir(
        exist_ok=True
    )

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=CLASS_NAMES
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    display.plot(
        xticks_rotation=45,
        values_format="d"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
            RESULTS_DIRECTORY
            / file_name
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to: "
        f"{output_path}"
    )


# --------------------------------
# Model Comparison Chart
# --------------------------------

def save_model_comparison(
        ml_accuracy,
        rule_accuracy,
        hybrid_accuracy
):
    """Creates a chart comparing all three approaches."""

    RESULTS_DIRECTORY.mkdir(
        exist_ok=True
    )

    model_names = [
        "ML Only",
        "Rules Only",
        "Hybrid"
    ]

    accuracies = [
        ml_accuracy,
        rule_accuracy,
        hybrid_accuracy
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        model_names,
        accuracies
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Classification Approach Comparison"
    )

    for bar, accuracy in zip(
            bars,
            accuracies
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{accuracy:.2%}",
            ha="center"
        )

    plt.tight_layout()

    output_path = (
            RESULTS_DIRECTORY
            / "model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Model comparison saved to: "
        f"{output_path}"
    )


# --------------------------------
# Sample Predictions
# --------------------------------

def show_sample_predictions(
        test_data,
        test_labels,
        predictions,
        sample_count=10
):
    """Prints example predictions from the hybrid system."""

    print(
        "\nSample Predictions"
    )

    print("=" * 50)

    sample_count = min(
        sample_count,
        len(test_data)
    )

    for index in range(
            sample_count
    ):
        problem = (
            test_data.iloc[index]["problem"]
        )

        student_answer = (
            test_data.iloc[index][
                "student_answer"
            ]
        )

        print(
            f"\nProblem: "
            f"{problem}"
        )

        print(
            f"Student Answer: "
            f"{student_answer}"
        )

        print(
            f"Actual: "
            f"{test_labels.iloc[index]}"
        )

        print(
            f"Predicted: "
            f"{predictions[index]}"
        )

        print(
            "-" * 50
        )


# --------------------------------
# Main Program
# --------------------------------

def main():
    """Runs the complete training and evaluation pipeline."""

    print(
        "Loading dataset..."
    )

    train_data, test_data = (
        load_dataset()
    )

    print(
        f"Training samples: "
        f"{len(train_data)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_data)}"
    )

    (
        vectorizer,
        train_features,
        test_features
    ) = prepare_features(
        train_data,
        test_data
    )

    train_labels = (
        train_data["label"]
    )

    test_labels = (
        test_data["label"]
    )

    print(
        "\nTraining Logistic Regression "
        "classifier..."
    )

    model = train_classifier(
        train_features,
        train_labels
    )

    # --------------------------------
    # Rule-Only Evaluation
    # --------------------------------

    print(
        "\nGenerating rule-only "
        "predictions..."
    )

    rule_predictions = (
        make_rule_predictions(
            test_data
        )
    )

    (
        rule_accuracy,
        rule_coverage,
        rule_match_accuracy
    ) = evaluate_rule_system(
        test_labels,
        rule_predictions
    )

    # --------------------------------
    # ML-Only Evaluation
    # --------------------------------

    print(
        "\nGenerating ML-only "
        "predictions..."
    )

    ml_predictions = (
        model.predict(
            test_features
        )
    )

    ml_accuracy = evaluate_model(
        test_labels,
        ml_predictions,
        "ML-Only Model"
    )

    save_confusion_matrix(
        test_labels,
        ml_predictions,
        "ML-Only Model",
        "ml_confusion_matrix.png"
    )

    # --------------------------------
    # Hybrid Evaluation
    # --------------------------------

    print(
        "\nGenerating hybrid "
        "predictions..."
    )

    hybrid_predictions = (
        make_hybrid_predictions(
            test_data,
            test_features,
            model
        )
    )

    hybrid_accuracy = evaluate_model(
        test_labels,
        hybrid_predictions,
        "Hybrid Model"
    )

    save_confusion_matrix(
        test_labels,
        hybrid_predictions,
        "Hybrid Model",
        "hybrid_confusion_matrix.png"
    )

    # --------------------------------
    # Model Comparison
    # --------------------------------

    print(
        "\nModel Comparison"
    )

    print("=" * 50)

    print(
        f"ML-Only Accuracy:    "
        f"{ml_accuracy:.4f}"
    )

    print(
        f"Rules-Only Accuracy: "
        f"{rule_accuracy:.4f}"
    )

    print(
        f"Hybrid Accuracy:     "
        f"{hybrid_accuracy:.4f}"
    )

    print(
        f"Rule Coverage:       "
        f"{rule_coverage:.4f}"
    )

    print(
        f"Rule Match Accuracy: "
        f"{rule_match_accuracy:.4f}"
    )

    ml_improvement = (
            hybrid_accuracy
            - ml_accuracy
    )

    rule_improvement = (
            hybrid_accuracy
            - rule_accuracy
    )

    print(
        f"\nHybrid vs. ML Only:    "
        f"{ml_improvement:+.4f}"
    )

    print(
        f"Hybrid vs. Rules Only: "
        f"{rule_improvement:+.4f}"
    )

    # Create comparison visualization
    save_model_comparison(
        ml_accuracy,
        rule_accuracy,
        hybrid_accuracy
    )

    # --------------------------------
    # Sample Hybrid Predictions
    # --------------------------------

    show_sample_predictions(
        test_data,
        test_labels,
        hybrid_predictions
    )


if __name__ == "__main__":
    main()