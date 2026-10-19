"""clean_data.py

Load raw data, clean/transform it, and write a tidy CSV ready for modeling.
Replace the placeholder logic below with the project's actual cleaning steps.
"""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("raw_data/data.csv")
OUTPUT_DIR = Path("output")
OUTPUT_PATH = OUTPUT_DIR / "clean_data.csv"


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    # TODO: add project-specific feature engineering / transformations here
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_raw_data(RAW_DATA_PATH)
    df = clean(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
