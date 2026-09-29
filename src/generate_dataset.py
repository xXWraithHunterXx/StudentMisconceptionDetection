import random
from pathlib import Path

import pandas as pd


# Keeps dataset generation reproducible
RANDOM_SEED = 42

# Number of problem pairs generated for each misconception category
EXAMPLES_PER_CATEGORY = 120

# Percentage of the dataset used for training
TRAIN_RATIO = 0.8


def choose_random(options):
    """Returns one randomly selected option."""
    return random.choice(options)


def generate_distribution_examples():
    """Generates correct answers and distributive-property errors."""
    examples = []

    for _ in range(EXAMPLES_PER_CATEGORY):
        coefficient = random.randint(2, 9)
        constant = random.randint(1, 9)

        problem = f"{coefficient}(x+{constant})"
        correct_answer = f"{coefficient}x+{coefficient * constant}"

        examples.append([problem, correct_answer, "correct"])

        wrong_answers = [
            f"{coefficient}x+{constant}",
            f"{coefficient}(x+{constant})",
            f"{coefficient}x+{coefficient * constant}+{constant}",
            f"{coefficient}x"
        ]

        examples.append([
            problem,
            choose_random(wrong_answers),
            "distributive_error"
        ])

    return examples


def generate_like_terms_examples():
    """Generates correct answers and like-terms errors."""
    examples = []

    for _ in range(EXAMPLES_PER_CATEGORY):
        first_coefficient = random.randint(1, 9)
        second_coefficient = random.randint(1, 9)

        problem = f"{first_coefficient}x+{second_coefficient}x"
        correct_answer = f"{first_coefficient + second_coefficient}x"

        examples.append([problem, correct_answer, "correct"])

        wrong_answers = [
            str(first_coefficient + second_coefficient),
            f"{first_coefficient * second_coefficient}x",
            f"{first_coefficient + second_coefficient}x^2",
            f"x^{first_coefficient + second_coefficient}"
        ]

        examples.append([
            problem,
            choose_random(wrong_answers),
            "like_terms_error"
        ])

    return examples


def generate_sign_examples():
    """Generates correct answers and sign-manipulation errors."""
    examples = []

    for _ in range(EXAMPLES_PER_CATEGORY):
        first_number = random.randint(5, 20)
        second_number = random.randint(1, 10)

        problem = f"{first_number}-(-{second_number})"
        correct_answer = str(first_number + second_number)

        examples.append([problem, correct_answer, "correct"])

        wrong_answers = [
            str(first_number - second_number),
            f"-{first_number + second_number}",
            str(first_number)
        ]

        random_wrong_answer = str(random.randint(0, 30))

        while random_wrong_answer == correct_answer:
            random_wrong_answer = str(random.randint(0, 30))

        wrong_answers.append(random_wrong_answer)

        examples.append([
            problem,
            choose_random(wrong_answers),
            "sign_error"
        ])

    return examples


def generate_exponent_examples():
    """Generates correct answers and basic exponent errors."""
    examples = []

    for _ in range(EXAMPLES_PER_CATEGORY):
        problem = "x*x"
        correct_answer = "x^2"

        examples.append([problem, correct_answer, "correct"])

        wrong_answers = [
            "2x",
            "x",
            "x^1",
            "xx"
        ]

        examples.append([
            problem,
            choose_random(wrong_answers),
            "exponent_error"
        ])

    return examples


def create_dataset():
    """Creates and returns the complete misconception dataset."""
    random.seed(RANDOM_SEED)

    examples = []

    examples.extend(generate_distribution_examples())
    examples.extend(generate_like_terms_examples())
    examples.extend(generate_sign_examples())
    examples.extend(generate_exponent_examples())

    dataset = pd.DataFrame(
        examples,
        columns=["problem", "student_answer", "label"]
    )

    # Shuffle reproducibly
    dataset = dataset.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    return dataset


def save_dataset(dataset):
    """Splits the dataset and saves the resulting CSV files."""
    train_size = int(len(dataset) * TRAIN_RATIO)

    train_data = dataset.iloc[:train_size]
    test_data = dataset.iloc[train_size:]

    # Finds the repository root regardless of working directory
    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"

    data_directory.mkdir(exist_ok=True)

    train_path = data_directory / "train.csv"
    test_path = data_directory / "test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print("Dataset generated successfully.")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Training data saved to: {train_path}")
    print(f"Testing data saved to: {test_path}")


def main():
    dataset = create_dataset()
    save_dataset(dataset)


if __name__ == "__main__":
    main()