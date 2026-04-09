import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_readmission_data(input_path: str, output_path: str) -> pd.DataFrame:

    df = pd.read_csv(input_path).copy()

  
    if "label" not in df.columns:
        raise ValueError("Input file must contain a 'label' column.")


    if "patient_id" in df.columns:
        df = df.drop_duplicates(subset=["patient_id"]).copy()

    if "admission_date" in df.columns:
        df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")

    
        df["admission_month"] = df["admission_date"].dt.month

        # cyclical encoding for months into sine and cosine values
        df["admission_month_sin"] = np.sin(2 * np.pi * df["admission_month"] / 12)
        df["admission_month_cos"] = np.cos(2 * np.pi * df["admission_month"] / 12)

    # binning of age into a more understandable age group variable
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[18, 30, 50, 70, 95],
            labels=["18-30", "30-50", "50-70", "70+"],
            include_lowest=True,
        )

    if "season" in df.columns:
        season_map = {
            "Winter": 1,
            "Spring": 2,
            "Summer": 3,
            "Fall": 4,
        }
        season_num = df["season"].map(season_map)
        df["season_sin"] = np.sin(2 * np.pi * season_num / 4)
        df["season_cos"] = np.cos(2 * np.pi * season_num / 4)

  
    columns_to_drop = [
        "patient_id",              # identifier not necessary
        "admission_date",          # raw date, better to represent with month
        "admission_month",         # raw month replaced by cyclical version
        "season",                  # raw season replaced by cyclical version
        "readmission_risk_score",  # likely leakage / proxy target feature as it is calculated from the rest of the dataset
        "comorbidities_count"      # was found to have a high correlation with the medication_count and age variables
    ]
    existing_drop_cols = [c for c in columns_to_drop if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)


    # saving to csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the hospital readmission dataset for model training."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="hospital_readmission_dataset.csv",
        help="Path to the raw input CSV file.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="hospital_readmission_dataset_preprocessed.csv",
        help="Path to save the preprocessed CSV file.",
    )
    args = parser.parse_args()

    processed = preprocess_readmission_data(args.input_csv, args.output_csv)

    print("Preprocessing complete.")
    print(f"Saved file: {args.output_csv}")
    print(f"Rows: {processed.shape[0]}")
    print(f"Columns: {processed.shape[1]}")
    print(f"Class distribution:\n{processed['label'].value_counts(dropna=False).sort_index()}")
