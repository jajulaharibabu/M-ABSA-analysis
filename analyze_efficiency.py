from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "official_coursera"

OUTPUT_FILE = (
    RESULTS_DIR /
    "efficiency_training_time_summary.csv"
)

MODELS = [
    "baseline",
    "dynamic_adapter_4_average",
    "dynamic_adapter_4"
]

MODEL_NAMES = {
    "baseline": "XLM-RoBERTa baseline",
    "dynamic_adapter_4_average": (
        "Four-adapter average fusion"
    ),
    "dynamic_adapter_4": (
        "Four-adapter dynamic routing"
    )
}

SEEDS = [42, 43, 44]

rows = []

for model_type in MODELS:
    seconds_list = []

    for seed in SEEDS:
        file_path = (
            RESULTS_DIR /
            f"{model_type}_seed{seed}_overall_metrics.json"
        )

        if not file_path.exists():
            print("Missing:", file_path)
            continue

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            metrics = json.load(file)

        seconds_list.append(
            metrics["training_seconds"]
        )

    if not seconds_list:
        continue

    mean_seconds = float(
        np.mean(seconds_list)
    )

    std_seconds = float(
        np.std(
            seconds_list,
            ddof=1
        )
    ) if len(seconds_list) > 1 else 0.0

    rows.append({
        "model": MODEL_NAMES[model_type],
        "completed_seeds": len(seconds_list),
        "mean_training_seconds": mean_seconds,
        "std_training_seconds": std_seconds,
        "mean_training_minutes": (
            mean_seconds / 60
        ),
        "std_training_minutes": (
            std_seconds / 60
        )
    })

summary_df = pd.DataFrame(rows)

summary_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nTraining-time summary:")
print(
    summary_df.to_string(
        index=False
    )
)