def is_answer_correct(problem, student_answer):
    """Checks whether a student's answer is correct for a supported problem."""

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    try:
        # Sign problems: a-(-b)
        if "-(-" in problem:
            first_number = int(problem.split("-(-")[0])
            second_number = int(problem.split("-(-")[1].replace(")", ""))

            correct_answer = first_number + second_number
            return int(student_answer) == correct_answer

        # Distribution problems: a(x+b)
        if "(" in problem and "+" in problem:
            coefficient = int(problem.split("(")[0])
            constant = int(problem.split("+")[1].replace(")", ""))

            correct_answer = f"{coefficient}x+{coefficient * constant}"
            return student_answer == correct_answer

        # Like terms problems: ax+bx
        if "x+" in problem:
            parts = problem.split("x+")

            first_coefficient = int(parts[0])
            second_coefficient = int(parts[1].replace("x", ""))

            correct_answer = f"{first_coefficient + second_coefficient}x"
            return student_answer == correct_answer

        # Exponent problem: x*x
        if problem == "x*x":
            return student_answer == "x^2"

    except (ValueError, IndexError):
        return False

    return False


def has_distributive_error(problem, student_answer):
    """Detects common distributive-property mistakes."""

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    if "(" not in problem or "x" not in problem:
        return False

    # Example: 4(x+3) left unchanged
    if "(" in student_answer:
        return True

    # Example: 4x instead of 4x+12
    if "x" in student_answer and "+" not in student_answer:
        return True

    return False


def has_sign_error(problem, student_answer):
    """Detects common mistakes involving subtraction of negative values."""

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    if "-(-" not in problem:
        return False

    try:
        first_number = int(problem.split("-(-")[0])
        second_number = int(problem.split("-(-")[1].replace(")", ""))

        correct_answer = first_number + second_number

        common_wrong_answers = [
            first_number - second_number,
            -(first_number + second_number),
            first_number
        ]

        student_value = int(student_answer)

        if student_value == correct_answer:
            return False

        return student_value in common_wrong_answers

    except (ValueError, IndexError):
        return False


def has_like_terms_error(problem, student_answer):
    """Detects common mistakes when combining like terms."""

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    if "x+" not in problem:
        return False

    try:
        parts = problem.split("x+")

        first_coefficient = int(parts[0])
        second_coefficient = int(parts[1].replace("x", ""))

        correct_sum = first_coefficient + second_coefficient

        common_wrong_answers = [
            str(correct_sum),
            f"{first_coefficient * second_coefficient}x",
            f"{correct_sum}x^2",
            f"x^{correct_sum}"
        ]

        return student_answer in common_wrong_answers

    except (ValueError, IndexError):
        return False


def has_exponent_error(problem, student_answer):
    """Detects common errors when simplifying x*x."""

    problem = str(problem).strip()
    student_answer = str(student_answer).strip()

    if problem != "x*x":
        return False

    common_wrong_answers = [
        "2x",
        "x",
        "x^1",
        "xx"
    ]

    return student_answer in common_wrong_answers
if __name__ == "__main__":
    print("Testing correctness rules:")
    print("16-(-6), 22:", is_answer_correct("16-(-6)", "22"))
    print("6(x+9), 6x+54:", is_answer_correct("6(x+9)", "6x+54"))
    print("3x+2x, 5x:", is_answer_correct("3x+2x", "5x"))
    print("x*x, x^2:", is_answer_correct("x*x", "x^2"))

    print("\nTesting misconception rules:")
    print("Sign error:", has_sign_error("13-(-1)", "13"))
    print("Distribution error:", has_distributive_error("4(x+4)", "4x"))
    print("Like terms error:", has_like_terms_error("3x+3x", "9x"))
    print("Exponent error:", has_exponent_error("x*x", "x"))