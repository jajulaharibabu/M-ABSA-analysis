# Dynamic Adapter Routing for Multilingual Aspect-Category Detection

This repository contains the code, dataset-preparation materials, trained-model outputs, and evaluation results for the study:

**Dynamic Adapter Routing for Multilingual Aspect-Category Detection**

The study evaluates whether review-level dynamic routing over four bottleneck adapters improves multilingual multi-label aspect-category detection on the official Coursera partition of the M-ABSA dataset.

## Paper summary

The proposed model uses XLM-RoBERTa-base as a multilingual encoder and adds four bottleneck adapters. A gating network computes input-dependent routing weights from the CLS representation and combines the adapter outputs through a weighted sum.

The model is compared with:

- An XLM-RoBERTa baseline.
- A four-adapter average-fusion model.
- A four-adapter dynamic-routing model.

All models are trained on the official Coursera train, development, and test partitions using three random seeds: 42, 43, and 44.

## Dataset

The experiments use the official Coursera partition of M-ABSA:

- Paper: *M-ABSA: A Multilingual Dataset for Aspect-Based Sentiment Analysis* (Wu et al., 2025)
- Official repository: https://github.com/swaggy66/M-ABSA
- Hugging Face dataset: https://huggingface.co/datasets/Multilingual-NLP/M-ABSA

The reconstructed Coursera dataset contains:

- 45,688 usable reviews
- 21 languages
- A reconstructed 31-label vocabulary
- 26,790 training reviews
- 7,033 development reviews
- 11,865 test reviews

The official train, development, and test partitions are retained using the `official_split` field.

### Important note on labels

The official M-ABSA annotations use triplets of the form:

```text
[aspect term, aspect category, sentiment polarity]
```

In the reconstructed Coursera dataset, sentiment-polarity values were retained in the `category_list` field for a small subset of records. As a result, the 31-label vocabulary includes three polarity-related labels:

- `polarity negative`
- `polarity neutral`
- `polarity positive`

These labels represent sentiment-polarity information rather than course-related aspect categories. They are used consistently across all models, seeds, and evaluation settings.

## Repository structure

```text
.
├── code/
│   ├── train_official_coursera.py
│   ├── dynamic_adapter_module.py
│   ├── analyze_per_language_results.py
│   ├── analyze_efficiency.py
│   ├── measure_inference_latency.py
│   └── consolidated_results.py
├── data/
│   └── coursera_mabsa_multilingual_with_language.csv
├── outputs/
│   └── official_coursera/
│       ├── baseline_seed42/
│       ├── baseline_seed43/
│       ├── baseline_seed44/
│       ├── dynamic_adapter_4_average_seed42/
│       ├── dynamic_adapter_4_average_seed43/
│       ├── dynamic_adapter_4_average_seed44/
│       ├── dynamic_adapter_4_seed42/
│       ├── dynamic_adapter_4_seed43/
│       └── dynamic_adapter_4_seed44/
└── results/
    └── official_coursera/
        ├── *_overall_metrics.json
        ├── *_category_report.csv
        ├── *_per_language_test.csv
        ├── *_config.json
        └── routing_analysis/
```

## Requirements

The experiments were implemented using Python, PyTorch, Hugging Face Transformers, Hugging Face Datasets, and scikit-learn.

Install the main dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas numpy safetensors
```

The reported GPU experiments were run on an NVIDIA Tesla T4 GPU in Google Colab with FP16 mixed precision enabled.

## Dataset preparation

Place the official M-ABSA Coursera records in the `data/` directory and run the dataset-preparation script to construct:

```text
data/coursera_mabsa_multilingual_with_language.csv
```

The reconstructed dataset must contain the following columns:

```text
sentence
category_list
official_split
language
annotations_json
num_annotations
domain
num_labels
category_text
```

## Training

To train a model, run:

```bash
python code/train_official_coursera.py \
  --model baseline \
  --seed 42
```

Available model choices:

```text
baseline
static_adapter
dynamic_adapter_4
dynamic_adapter_4_average
```

Examples:

```bash
# XLM-RoBERTa baseline
python code/train_official_coursera.py --model baseline --seed 42

# Four-adapter average fusion
python code/train_official_coursera.py --model dynamic_adapter_4_average --seed 42

# Four-adapter dynamic routing
python code/train_official_coursera.py --model dynamic_adapter_4 --seed 42
```

To reproduce all main results, run each model with seeds 42, 43, and 44:

```bash
for model in baseline dynamic_adapter_4_average dynamic_adapter_4; do
  for seed in 42 43 44; do
    python code/train_official_coursera.py --model $model --seed $seed
  done
done
```

## Training configuration

| Component | Setting |
|---|---|
| Encoder | XLM-RoBERTa-base |
| Maximum sequence length | 128 |
| Labels | 31 |
| Adapters | 4 |
| Bottleneck size | 64 |
| Training batch size | 16 |
| Evaluation batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Dropout | 0.1 |
| Loss | Weighted binary cross-entropy with logits |
| Positive-class weight clipping | [1, 30] |
| Checkpoint selection | Lowest development loss |
| Threshold search | 0.05 to 0.95, step 0.01 |
| Threshold-selection data | Development set only |
| Optimiser | AdamW through Hugging Face Trainer |
| Warm-up | None |
| Precision | FP16 mixed precision on GPU |
| Seeds | 42, 43, 44 |

## Outputs

For every model-seed run, the training script saves:

```text
<model>_seed<seed>_overall_metrics.json
<model>_seed<seed>_category_report.csv
<model>_seed<seed>_per_language_test.csv
<model>_seed<seed>_config.json
```

Example:

```text
results/official_coursera/dynamic_adapter_4_seed42_overall_metrics.json
results/official_coursera/dynamic_adapter_4_seed42_category_report.csv
results/official_coursera/dynamic_adapter_4_seed42_per_language_test.csv
results/official_coursera/dynamic_adapter_4_seed42_config.json
```

## Main results

Results are reported as mean ± standard deviation across seeds 42, 43, and 44.

| Model | Micro-F1 | Subset accuracy | Macro-F1 |
|---|---:|---:|---:|
| XLM-RoBERTa baseline | 0.5132 ± 0.0205 | 0.2937 ± 0.0516 | 0.2625 ± 0.0082 |
| Four-adapter average fusion | 0.5145 ± 0.0110 | 0.2907 ± 0.0310 | 0.2539 ± 0.0047 |
| Four-adapter dynamic routing | 0.5261 ± 0.0145 | 0.3231 ± 0.0373 | 0.2531 ± 0.0096 |

Dynamic routing improves micro-F1 and subset accuracy over both comparison models in each of the three seeds. Macro-F1 does not improve.

## Checkpoints

The `outputs/official_coursera/` directory contains one folder per model-seed pair. Each folder contains Hugging Face Trainer checkpoints.

Because pretrained encoder checkpoints are large, checkpoints may be released separately through Zenodo, Hugging Face Hub, or institutional storage.

## Citation

If you use this code or the reconstructed dataset, please cite the M-ABSA paper:

```bibtex
@inproceedings{wu-etal-2025-mabsa,
  title = "{M-ABSA}: A Multilingual Dataset for Aspect-Based Sentiment Analysis",
  author = "Wu, ChengYan and
            Ma, Bolei and
            Liu, Yihong and
            Zhang, Zheyu and
            Deng, Ningyuan and
            Li, Yanshu and
            Chen, Baolan and
            Zhang, Yi and
            Xue, Yun and
            Plank, Barbara",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  year = "2025",
  pages = "2530--2557",
  publisher = "Association for Computational Linguistics"
}
```

## License

[Add the licence for your code, e.g., MIT or Apache 2.0.]

The M-ABSA dataset is subject to the licence and terms of use of its official repository.
