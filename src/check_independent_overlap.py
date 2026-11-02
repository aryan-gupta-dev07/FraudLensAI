from pathlib import Path
import pandas as pd

NEW_DATASET = Path("data/independent/Dataset.csv")
ORIGINAL_DATASET = Path("data/urlset_clean.csv")


def main():
    print("Loading datasets...")

    new_df = pd.read_csv(NEW_DATASET)
    original_df = pd.read_csv(ORIGINAL_DATASET)

    print("\n--- CHECKING URL COLUMNS ---")

    print("New dataset columns:", list(new_df.columns))
    print("Original dataset columns:", list(original_df.columns))

    if "url" not in new_df.columns:
        print("ERROR: URL column not found in new dataset.")
        return

    if "domain" not in original_df.columns:
        print("ERROR: Domain column not found in original dataset.")
        return

    new_urls = set(
        new_df["url"].dropna().astype(str).str.strip().str.lower()
    )

    original_urls = set(
        original_df["domain"].dropna().astype(str).str.strip().str.lower()
    )

    overlap = new_urls.intersection(original_urls)

    print("\n--- OVERLAP RESULTS ---")
    print("Unique URLs in new dataset:", len(new_urls))
    print("Unique URLs in original dataset:", len(original_urls))
    print("Exact overlapping URLs:", len(overlap))

    if new_urls:
        percentage = (len(overlap) / len(new_urls)) * 100
        print(f"Overlap percentage: {percentage:.4f}%")

    print("\nNo URLs were opened or visited.")
    print("Overlap check completed.")


if __name__ == "__main__":
    main()