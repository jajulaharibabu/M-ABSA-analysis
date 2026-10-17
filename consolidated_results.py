import os, glob
import pandas as pd

RESULTS_DIR = "/content/drive/MyDrive/ABSA_Project/results/official_coursera"
OUT_FILE = "/content/drive/MyDrive/ABSA_Project/results/consolidated_paper_results.csv"

all_rows = []

for path in glob.glob(os.path.join(RESULTS_DIR, "**", "*.csv"), recursive=True):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("Skipped:", path, "| Error:", e)
        continue

    rel_path = os.path.relpath(path, RESULTS_DIR)
    df["source_file"] = rel_path
    all_rows.append(df)

if not all_rows:
    raise ValueError("No readable CSV files found.")

combined = pd.concat(all_rows, ignore_index=True, sort=False)

# Keep only useful columns if they exist
useful_cols = [
    "model", "model_type", "seed", "language", "lang",
    "micro_f1", "subset_accuracy", "macro_f1",
    "accuracy", "f1", "precision", "recall",
    "parameters", "training_time_minutes",
    "milliseconds_per_sample", "samples_per_second",
    "source_file"
]

existing = [c for c in useful_cols if c in combined.columns]
combined = combined[existing]

combined.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print("Total rows:", len(combined))
print("Columns:", list(combined.columns))
print(combined.head(20))