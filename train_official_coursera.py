from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)

from dynamic_adapter_module import DynamicAdapter


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" /
    "coursera_mabsa_multilingual_with_language.csv"
)

RESULTS_DIR = ROOT / "results" / "official_coursera"
OUTPUT_ROOT = ROOT / "outputs" / "official_coursera"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
BOTTLENECK_SIZE = 64

THRESHOLD_START = 0.05
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.01


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=[
        "baseline",
        "static_adapter",
        "dynamic_adapter_4",
        "dynamic_adapter_4_average"
    ]
)

parser.add_argument(
    "--seed",
    type=int,
    required=True
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE
)

parser.add_argument(
    "--epochs",
    type=int,
    default=NUM_EPOCHS
)

args = parser.parse_args()

MODEL_TYPE = args.model
SEED = args.seed
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

set_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device_name = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

print("Device:", device_name)
print("Model:", MODEL_TYPE)
print("Seed:", SEED)
print("Dataset:", DATA_FILE)


df = pd.read_csv(
    DATA_FILE,
    keep_default_na=False
)

required_columns = {
    "sentence",
    "category_list",
    "official_split",
    "language"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "Missing columns: "
        f"{sorted(missing_columns)}"
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

category2id = {
    category: index
    for index, category in enumerate(all_categories)
}

num_categories = len(all_categories)

print("Number of categories:", num_categories)
print("Categories:", all_categories)


def create_label_vector(category_list):
    labels = np.zeros(
        num_categories,
        dtype=np.float32
    )

    for category in category_list:
        category = str(category).strip()

        if category in category2id:
            labels[category2id[category]] = 1.0

    return labels


df["labels"] = df["category_list"].apply(
    create_label_vector
)

train_df = df[
    df["official_split"] == "train"
].reset_index(drop=True)

dev_df = df[
    df["official_split"] == "dev"
].reset_index(drop=True)

test_df = df[
    df["official_split"] == "test"
].reset_index(drop=True)

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Dev:", len(dev_df))
print("Test:", len(test_df))

train_matrix = np.stack(
    train_df["labels"].to_numpy()
)

positive_counts = train_matrix.sum(axis=0)

negative_counts = (
    len(train_matrix) - positive_counts
)

positive_weights = np.divide(
    negative_counts,
    positive_counts,
    out=np.ones_like(
        negative_counts,
        dtype=np.float32
    ),
    where=positive_counts > 0
)

positive_weights = np.clip(
    positive_weights,
    a_min=1.0,
    a_max=30.0
)

pos_weight_tensor = torch.tensor(
    positive_weights,
    dtype=torch.float32
)

print("\nPositive-class weight range:")
print(
    float(positive_weights.min()),
    "to",
    float(positive_weights.max())
)


def dataframe_to_dataset(dataframe):
    return Dataset.from_pandas(
        dataframe[
            [
                "sentence",
                "labels"
            ]
        ],
        preserve_index=False
    )


train_dataset = dataframe_to_dataset(
    train_df
)

dev_dataset = dataframe_to_dataset(
    dev_df
)

test_dataset = dataframe_to_dataset(
    test_df
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


train_dataset = train_dataset.map(
    tokenize_batch,
    batched=True
)

dev_dataset = dev_dataset.map(
    tokenize_batch,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_batch,
    batched=True
)

format_columns = [
    "input_ids",
    "attention_mask",
    "labels"
]

train_dataset.set_format(
    "torch",
    columns=format_columns
)

dev_dataset.set_format(
    "torch",
    columns=format_columns
)

test_dataset.set_format(
    "torch",
    columns=format_columns
)


class BaseMultilabelModel(nn.Module):

    def calculate_loss(
        self,
        logits,
        labels
    ):
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor.to(
                logits.device
            )
        )

        return loss_function(
            logits,
            labels.float()
        )


class BaselineModel(
    BaseMultilabelModel
):

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

        loss = None

        if labels is not None:
            loss = self.calculate_loss(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }


class StaticAdapterModel(
    BaseMultilabelModel
):

    def __init__(
        self,
        model_name,
        num_labels,
        bottleneck_size=BOTTLENECK_SIZE
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.encoder.config.hidden_size

        self.adapter = DynamicAdapter(
            hidden_size,
            num_adapters=1,
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

        adapted = self.adapter(
            hidden_states
        )

        pooled = adapted[:, 0]

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        loss = None

        if labels is not None:
            loss = self.calculate_loss(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }


class DynamicAdapterModel(
    BaseMultilabelModel
):

    def __init__(
        self,
        model_name,
        num_labels,
        num_adapters=4,
        bottleneck_size=BOTTLENECK_SIZE
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

        loss = None

        if labels is not None:
            loss = self.calculate_loss(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }


class DynamicAdapterAverageModel(
    BaseMultilabelModel
):

    def __init__(
        self,
        model_name,
        num_labels,
        num_adapters=4,
        bottleneck_size=BOTTLENECK_SIZE
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

        loss = None

        if labels is not None:
            loss = self.calculate_loss(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }


if MODEL_TYPE == "baseline":
    model = BaselineModel(
        MODEL_NAME,
        num_categories
    )

elif MODEL_TYPE == "static_adapter":
    model = StaticAdapterModel(
        MODEL_NAME,
        num_categories
    )

elif MODEL_TYPE == "dynamic_adapter_4":
    model = DynamicAdapterModel(
        MODEL_NAME,
        num_categories,
        num_adapters=4
    )

elif MODEL_TYPE == "dynamic_adapter_4_average":
    model = DynamicAdapterAverageModel(
        MODEL_NAME,
        num_categories,
        num_adapters=4
    )

else:
    raise ValueError(
        f"Unsupported model type: {MODEL_TYPE}"
    )

output_dir = OUTPUT_ROOT / (
    f"{MODEL_TYPE}_seed{SEED}"
)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2
)


def compute_metrics(prediction):
    logits = prediction.predictions
    true_labels = prediction.label_ids

    probabilities = torch.sigmoid(
        torch.tensor(logits)
    ).numpy()

    predicted_labels = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "micro_f1_at_050": f1_score(
            true_labels,
            predicted_labels,
            average="micro",
            zero_division=0
        ),
        "macro_f1_at_050": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )
    }


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start_time = time.time()

trainer.train()

training_seconds = time.time() - start_time

dev_prediction = trainer.predict(
    dev_dataset
)

dev_logits = dev_prediction.predictions
dev_true_labels = dev_prediction.label_ids

dev_probabilities = torch.sigmoid(
    torch.tensor(dev_logits)
).numpy()

thresholds = np.arange(
    THRESHOLD_START,
    THRESHOLD_END + 1e-9,
    THRESHOLD_STEP
)

best_threshold = None
best_balanced_score = -1.0
best_dev_micro_f1 = 0.0
best_dev_macro_f1 = 0.0

for threshold in thresholds:
    dev_predicted_labels = (
        dev_probabilities >= threshold
    ).astype(int)

    dev_micro_f1 = f1_score(
        dev_true_labels,
        dev_predicted_labels,
        average="micro",
        zero_division=0
    )

    dev_macro_f1 = f1_score(
        dev_true_labels,
        dev_predicted_labels,
        average="macro",
        zero_division=0
    )

    balanced_score = (
        dev_micro_f1 +
        dev_macro_f1
    ) / 2

    if balanced_score > best_balanced_score:
        best_threshold = float(threshold)
        best_balanced_score = float(
            balanced_score
        )
        best_dev_micro_f1 = float(
            dev_micro_f1
        )
        best_dev_macro_f1 = float(
            dev_macro_f1
        )

print("\nBest dev threshold:", best_threshold)
print("Best dev balanced score:", best_balanced_score)
print("Best dev micro-F1:", best_dev_micro_f1)
print("Best dev macro-F1:", best_dev_macro_f1)

test_prediction = trainer.predict(
    test_dataset
)

test_logits = test_prediction.predictions
test_true_labels = test_prediction.label_ids

test_probabilities = torch.sigmoid(
    torch.tensor(test_logits)
).numpy()

test_predicted_labels = (
    test_probabilities >= best_threshold
).astype(int)

test_micro_f1 = f1_score(
    test_true_labels,
    test_predicted_labels,
    average="micro",
    zero_division=0
)

test_macro_f1 = f1_score(
    test_true_labels,
    test_predicted_labels,
    average="macro",
    zero_division=0
)

test_subset_accuracy = accuracy_score(
    test_true_labels,
    test_predicted_labels
)

print("\nFinal test results")
print("Threshold:", best_threshold)
print("Test micro-F1:", test_micro_f1)
print("Test macro-F1:", test_macro_f1)
print("Test subset accuracy:", test_subset_accuracy)

overall_metrics = {
    "model": MODEL_TYPE,
    "seed": SEED,
    "device": device_name,
    "dataset": "M-ABSA Coursera",
    "domain": "coursera",
    "num_languages": int(
        df["language"].nunique()
    ),
    "num_categories": num_categories,
    "train_size": len(train_df),
    "dev_size": len(dev_df),
    "test_size": len(test_df),
    "positive_weight_min": float(
        positive_weights.min()
    ),
    "positive_weight_max": float(
        positive_weights.max()
    ),
    "best_dev_threshold": best_threshold,
    "best_dev_balanced_score": best_balanced_score,
    "best_dev_micro_f1": best_dev_micro_f1,
    "best_dev_macro_f1": best_dev_macro_f1,
    "test_micro_f1": float(test_micro_f1),
    "test_macro_f1": float(test_macro_f1),
    "test_subset_accuracy": float(
        test_subset_accuracy
    ),
    "training_seconds": float(
        training_seconds
    )
}

overall_file = RESULTS_DIR / (
    f"{MODEL_TYPE}_seed{SEED}_overall_metrics.json"
)

with open(
    overall_file,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        overall_metrics,
        file,
        indent=2,
        ensure_ascii=False
    )

report = classification_report(
    test_true_labels,
    test_predicted_labels,
    target_names=all_categories,
    output_dict=True,
    zero_division=0
)

category_rows = []

for category in all_categories:
    category_metrics = report.get(
        category,
        {}
    )

    category_rows.append({
        "category": category,
        "precision": category_metrics.get(
            "precision",
            0.0
        ),
        "recall": category_metrics.get(
            "recall",
            0.0
        ),
        "f1": category_metrics.get(
            "f1-score",
            0.0
        ),
        "support": category_metrics.get(
            "support",
            0
        )
    })

category_report_file = RESULTS_DIR / (
    f"{MODEL_TYPE}_seed{SEED}_category_report.csv"
)

pd.DataFrame(category_rows).to_csv(
    category_report_file,
    index=False,
    encoding="utf-8-sig"
)

language_rows = []

for language in sorted(
    test_df["language"].unique()
):
    language_mask = (
        test_df["language"].to_numpy() ==
        language
    )

    language_true = test_true_labels[
        language_mask
    ]

    language_predicted = test_predicted_labels[
        language_mask
    ]

    language_micro_f1 = f1_score(
        language_true,
        language_predicted,
        average="micro",
        zero_division=0
    )

    language_macro_f1 = f1_score(
        language_true,
        language_predicted,
        average="macro",
        zero_division=0
    )

    language_subset_accuracy = accuracy_score(
        language_true,
        language_predicted
    )

    language_rows.append({
        "language": language,
        "test_samples": int(
            language_mask.sum()
        ),
        "micro_f1": language_micro_f1,
        "macro_f1": language_macro_f1,
        "subset_accuracy": language_subset_accuracy
    })

language_report_file = RESULTS_DIR / (
    f"{MODEL_TYPE}_seed{SEED}_per_language_test.csv"
)

pd.DataFrame(language_rows).to_csv(
    language_report_file,
    index=False,
    encoding="utf-8-sig"
)

config_file = RESULTS_DIR / (
    f"{MODEL_TYPE}_seed{SEED}_config.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "model_name": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "bottleneck_size": BOTTLENECK_SIZE,
            "loss": "BCEWithLogitsLoss with clipped pos_weight",
            "threshold_selection": {
                "dataset": "development split only",
                "objective": "mean of micro-F1 and macro-F1",
                "start": THRESHOLD_START,
                "end": THRESHOLD_END,
                "step": THRESHOLD_STEP
            }
        },
        file,
        indent=2
    )

print("\nSaved files:")
print(overall_file)
print(category_report_file)
print(language_report_file)
print(config_file)