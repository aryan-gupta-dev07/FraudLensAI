from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/independent/Dataset.csv")


def main():
    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found.")
        return

    print("Loading independent dataset...")

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as error:
        print("ERROR: Could not read dataset.")
        print(type(error).__name__)
        return

    print("\n--- DATASET OVERVIEW ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\n--- COLUMN NAMES ---")
    for column in df.columns:
        print("-", column)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().to_string())

    print("\n--- DUPLICATE ROWS ---")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n--- POSSIBLE URL COLUMNS ---")
    for column in df.columns:
        if any(word in column.lower() for word in ["url", "link", "domain"]):
            print("-", column)

    print("\n--- POSSIBLE LABEL COLUMNS ---")
    for column in df.columns:
        if any(word in column.lower() for word in ["label", "class", "target"]):
            print("-", column)
            print(df[column].value_counts(dropna=False).to_string())

    print("\nInspection completed.")
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()