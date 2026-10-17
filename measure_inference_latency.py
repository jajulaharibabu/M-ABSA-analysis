from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from dynamic_adapter_module import DynamicAdapter


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" /
    "coursera_mabsa_multilingual_with_language.csv"
)

OUTPUT_ROOT = ROOT / "outputs" / "official_coursera"

RESULTS_DIR = ROOT / "results" / "official_coursera"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 128
BATCH_SIZE = 16
SEED = 42

WARMUP_BATCHES = 10
MEASURE_BATCHES = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


df = pd.read_csv(
    DATA_FILE,
    keep_default_na=False
)

df = df[
    df["sentence"].str.strip() != ""
].copy()

df["category_list"] = df["category_list"].apply(
    json.loads
)

all_categories = sorted(
    {
        str(category).strip()
        for category_list in df["category_list"]
        for category in category_list
        if str(category).strip()
    }
)

num_categories = len(all_categories)

test_df = df[
    df["official_split"] == "test"
].reset_index(drop=True)

test_dataset = Dataset.from_pandas(
    test_df[["sentence"]],
    preserve_index=False
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )


test_dataset = test_dataset.map(
    tokenize_batch,
    batched=True
)

test_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask"
    ]
)


class BaselineModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return {
            "logits": logits
        }


class DynamicAdapterModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels,
        num_adapters=4,
        bottleneck_size=64
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.dynamic_adapter = DynamicAdapter(
            hidden_size,
            num_adapters=num_adapters,
            bottleneck_size=bottleneck_size
        )

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        adapted = self.dynamic_adapter(
            hidden_states
        )

        pooled = adapted[:, 0]

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return {
            "logits": logits
        }


class DynamicAdapterAverageModel(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels,
        num_adapters=4,
        bottleneck_size=64
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.adapters = nn.ModuleList(
            [
                DynamicAdapter(
                    hidden_size,
                    num_adapters=1,
                    bottleneck_size=bottleneck_size
                )
                for _ in range(num_adapters)
            ]
        )

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        adapter_outputs = [
            adapter(hidden_states)
            for adapter in self.adapters
        ]

        adapted = torch.stack(
            adapter_outputs,
            dim=0
        ).mean(dim=0)

        pooled = adapted[:, 0]

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return {
            "logits": logits
        }


def find_last_checkpoint(model_type):
    model_dir = OUTPUT_ROOT / (
        f"{model_type}_seed{SEED}"
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model output directory not found: "
            f"{model_dir}"
        )

    checkpoints = [
        folder
        for folder in model_dir.iterdir()
        if folder.is_dir()
        and folder.name.startswith("checkpoint-")
    ]

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint folders found in: "
            f"{model_dir}"
        )

    checkpoints.sort(
        key=lambda folder: int(
            folder.name.replace(
                "checkpoint-",
                ""
            )
        )
    )

    return checkpoints[-1]


def load_checkpoint_weights(
    model,
    checkpoint_dir
):
    safe_file = checkpoint_dir / "model.safetensors"

    pytorch_file = checkpoint_dir / "pytorch_model.bin"

    if safe_file.exists():
        from safetensors.torch import load_file

        state_dict = load_file(
            safe_file,
            device="cpu"
        )

    elif pytorch_file.exists():
        state_dict = torch.load(
            pytorch_file,
            map_location="cpu"
        )

    else:
        raise FileNotFoundError(
            "No model.safetensors or "
            "pytorch_model.bin found in "
            f"{checkpoint_dir}"
        )

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("\nLoaded checkpoint:")
    print(checkpoint_dir)

    if missing_keys:
        print(
            "Missing keys:",
            len(missing_keys)
        )

    if unexpected_keys:
        print(
            "Unexpected keys:",
            len(unexpected_keys)
        )


def create_model(model_type):
    if model_type == "baseline":
        return BaselineModel(
            MODEL_NAME,
            num_categories
        )

    if model_type == "dynamic_adapter_4_average":
        return DynamicAdapterAverageModel(
            MODEL_NAME,
            num_categories,
            num_adapters=4
        )

    if model_type == "dynamic_adapter_4":
        return DynamicAdapterModel(
            MODEL_NAME,
            num_categories,
            num_adapters=4
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def measure_latency(
    model,
    dataset,
    batch_size,
    warmup_batches,
    measure_batches
):
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    model = model.to(DEVICE)
    model.eval()

    batches = list(loader)

    total_available_batches = len(batches)

    usable_measure_batches = min(
        measure_batches,
        total_available_batches
    )

    warmup_count = min(
        warmup_batches,
        total_available_batches
    )

    with torch.no_grad():
        for batch_index in range(warmup_count):
            batch = {
                key: value.to(
                    DEVICE,
                    non_blocking=True
                )
                for key, value in batches[batch_index].items()
            }

            _ = model(**batch)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    measured_samples = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index in range(
            usable_measure_batches
        ):
            batch = {
                key: value.to(
                    DEVICE,
                    non_blocking=True
                )
                for key, value in batches[batch_index].items()
            }

            _ = model(**batch)

            measured_samples += batch[
                "input_ids"
            ].size(0)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    milliseconds_per_sample = (
        elapsed_seconds * 1000 /
        measured_samples
    )

    batches_per_second = (
        usable_measure_batches /
        elapsed_seconds
    )

    samples_per_second = (
        measured_samples /
        elapsed_seconds
    )

    return {
        "measured_batches": usable_measure_batches,
        "measured_samples": measured_samples,
        "elapsed_seconds": elapsed_seconds,
        "milliseconds_per_sample": (
            milliseconds_per_sample
        ),
        "samples_per_second": (
            samples_per_second
        ),
        "batches_per_second": (
            batches_per_second
        )
    }


MODEL_TYPES = [
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

results = []

for model_type in MODEL_TYPES:
    print("\n" + "=" * 70)
    print("Measuring:", MODEL_NAMES[model_type])
    print("=" * 70)

    model = create_model(
        model_type
    )

    checkpoint_dir = find_last_checkpoint(
        model_type
    )

    load_checkpoint_weights(
        model,
        checkpoint_dir
    )

    latency_result = measure_latency(
        model=model,
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        warmup_batches=WARMUP_BATCHES,
        measure_batches=MEASURE_BATCHES
    )

    latency_result["model"] = (
        MODEL_NAMES[model_type]
    )

    latency_result["model_type"] = model_type

    latency_result["seed"] = SEED

    results.append(
        latency_result
    )

    print("\nLatency result:")
    print(
        json.dumps(
            latency_result,
            indent=2
        )
    )

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results)

output_file = RESULTS_DIR / (
    "inference_latency_seed42.csv"
)

results_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(output_file)

print("\nInference-latency summary:")
print(
    results_df[
        [
            "model",
            "seed",
            "measured_samples",
            "milliseconds_per_sample",
            "samples_per_second",
            "batches_per_second"
        ]
    ].to_string(
        index=False
    )
)