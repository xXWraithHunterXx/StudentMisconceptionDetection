from train_model import (
    load_dataset,
    prepare_features,
    train_classifier,
    predict_misconception
)


def get_feedback(prediction):
    """Returns educational feedback for a predicted misconception."""

    feedback_messages = {
        "correct": (
            "Correct! The answer was simplified properly."
        ),

        "distributive_error": (
            "Check the distributive property. Multiply the "
            "coefficient by every term inside the parentheses."
        ),

        "sign_error": (
            "Check the positive and negative signs carefully. "
            "Remember that subtracting a negative value is "
            "equivalent to adding it."
        ),

        "like_terms_error": (
            "Check how the like terms were combined. "
            "Add or subtract their coefficients while keeping "
            "the variable and exponent unchanged."
        ),

        "exponent_error": (
            "Check the exponent rule. When multiplying the same "
            "variable, add the exponents."
        )
    }

    return feedback_messages.get(
        prediction,
        "No specific feedback is available for this response."
    )


def main():
    """Runs the interactive misconception detection demo."""

    print("=" * 55)
    print("AI Student Misconception Detector")
    print("=" * 55)

    print("\nLoading model...")

    train_data, test_data = load_dataset()

    (
        vectorizer,
        train_features,
        test_features
    ) = prepare_features(
        train_data,
        test_data
    )

    train_labels = train_data["label"]

    model = train_classifier(
        train_features,
        train_labels
    )

    print("Model ready.")

    print("\nSupported problem formats:")
    print("  Distribution:  4(x+3)")
    print("  Like terms:    3x+2x")
    print("  Sign errors:   8-(-4)")
    print("  Exponents:     x*x")

    while True:
        print("\n" + "-" * 55)

        problem = input(
            "Enter an algebra problem "
            "(or 'quit' to exit): "
        ).strip()

        if problem.lower() == "quit":
            print("\nExiting misconception detector.")
            break

        student_answer = input(
            "Enter the student's answer: "
        ).strip()

        input_text = (
                problem
                + " "
                + student_answer
        )

        ml_features = vectorizer.transform(
            [input_text]
        )

        prediction = predict_misconception(
            problem,
            student_answer,
            ml_features,
            model
        )

        feedback = get_feedback(
            prediction
        )

        print(
            f"\nClassification: "
            f"{prediction}"
        )

        print(
            f"Feedback: "
            f"{feedback}"
        )


if __name__ == "__main__":
    main()