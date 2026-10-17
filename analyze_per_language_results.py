from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "official_coursera"

OUTPUT_FILE = (
    RESULTS_DIR /
    "per_language_baseline_vs_dynamic_summary.csv"
)

SEEDS = [42, 43, 44]

BASELINE_MODEL = "baseline"
DYNAMIC_MODEL = "dynamic_adapter_4"

metrics = [
    "micro_f1",
    "macro_f1",
    "subset_accuracy"
]

all_model_frames = []

for model_name in [
    BASELINE_MODEL,
    DYNAMIC_MODEL
]:
    for seed in SEEDS:
        file_path = (
            RESULTS_DIR /
            f"{model_name}_seed{seed}_per_language_test.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        df = pd.read_csv(
            file_path
        )

        df["model"] = model_name
        df["seed"] = seed

        all_model_frames.append(df)

results_df = pd.concat(
    all_model_frames,
    ignore_index=True
)

summary_rows = []

languages = sorted(
    results_df["language"].unique()
)

for language in languages:
    language_df = results_df[
        results_df["language"] == language
    ]

    baseline_df = language_df[
        language_df["model"] == BASELINE_MODEL
    ]

    dynamic_df = language_df[
        language_df["model"] == DYNAMIC_MODEL
    ]

    row = {
        "language": language,
        "test_samples": int(
            baseline_df["test_samples"].iloc[0]
        )
    }

    for metric in metrics:
        baseline_values = baseline_df[
            metric
        ].to_numpy()

        dynamic_values = dynamic_df[
            metric
        ].to_numpy()

        row[f"baseline_{metric}_mean"] = float(
            baseline_values.mean()
        )

        row[f"baseline_{metric}_std"] = float(
            baseline_values.std(ddof=1)
        )

        row[f"dynamic_{metric}_mean"] = float(
            dynamic_values.mean()
        )

        row[f"dynamic_{metric}_std"] = float(
            dynamic_values.std(ddof=1)
        )

        row[f"dynamic_minus_baseline_{metric}"] = float(
            dynamic_values.mean() -
            baseline_values.mean()
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df = summary_df.sort_values(
    "dynamic_minus_baseline_micro_f1",
    ascending=False
).reset_index(drop=True)

summary_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nPer-language micro-F1 comparison:")
print(
    summary_df[
        [
            "language",
            "test_samples",
            "baseline_micro_f1_mean",
            "dynamic_micro_f1_mean",
            "dynamic_minus_baseline_micro_f1",
            "baseline_macro_f1_mean",
            "dynamic_macro_f1_mean",
            "dynamic_minus_baseline_macro_f1",
            "baseline_subset_accuracy_mean",
            "dynamic_subset_accuracy_mean",
            "dynamic_minus_baseline_subset_accuracy"
        ]
    ].to_string(
        index=False
    )
)

micro_gain = summary_df[
    "dynamic_minus_baseline_micro_f1"
]

macro_gain = summary_df[
    "dynamic_minus_baseline_macro_f1"
]

subset_gain = summary_df[
    "dynamic_minus_baseline_subset_accuracy"
]

print("\nSummary:")
print(
    "Languages improved in micro-F1:",
    int((micro_gain > 0).sum()),
    "/",
    len(summary_df)
)

print(
    "Languages improved in macro-F1:",
    int((macro_gain > 0).sum()),
    "/",
    len(summary_df)
)

print(
    "Languages improved in subset accuracy:",
    int((subset_gain > 0).sum()),
    "/",
    len(summary_df)
)

print("\nTop 5 micro-F1 gains:")
print(
    summary_df[
        [
            "language",
            "baseline_micro_f1_mean",
            "dynamic_micro_f1_mean",
            "dynamic_minus_baseline_micro_f1"
        ]
    ].head(5).to_string(
        index=False
    )
)

print("\nBottom 5 micro-F1 gains:")
print(
    summary_df[
        [
            "language",
            "baseline_micro_f1_mean",
            "dynamic_micro_f1_mean",
            "dynamic_minus_baseline_micro_f1"
        ]
    ].tail(5).to_string(
        index=False
    )
)