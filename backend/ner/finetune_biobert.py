"""
ClearScript — BioBERT Fine-Tuning Script

Fine-tunes the d4data/biomedical-ner-all model on annotated Indian
medical lab reports for improved accuracy on local formats.

Supports input from Label Studio exports in CoNLL 2003 format.

Usage:
    python -m backend.ner.finetune_biobert --data annotation/exported_annotations.conll
    python -m backend.ner.finetune_biobert --help

Requirements (install before running):
    pip install datasets seqeval accelerate
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Ensure project root is on sys.path ────────────────────────────────────────
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.config import BIOBERT_NER_MODEL


# ── Default hyperparameters ───────────────────────────────────────────────────

DEFAULTS = {
    "base_model": BIOBERT_NER_MODEL,
    "output_dir": os.path.join(_project_root, "backend", "models", "finetuned-biobert-ner"),
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_seq_length": 512,
    "val_split": 0.2,
    "seed": 42,
    "early_stopping_patience": 3,
}


# ── CoNLL 2003 parser ────────────────────────────────────────────────────────

def parse_conll_file(filepath: str) -> tuple[list[list[str]], list[list[str]]]:
    """
    Parse a CoNLL 2003 format file into token and label lists.

    Each sentence is separated by a blank line. Each line contains:
        token label
    or with additional columns (POS, chunk):
        token POS chunk label

    Returns:
        (list of token lists, list of label lists)
    """
    all_tokens: list[list[str]] = []
    all_labels: list[list[str]] = []
    current_tokens: list[str] = []
    current_labels: list[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            # Skip document boundaries
            if line.startswith("-DOCSTART-") or line.startswith("# "):
                continue

            if line == "":
                # End of sentence
                if current_tokens:
                    all_tokens.append(current_tokens)
                    all_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]  # Label is always the last column
                current_tokens.append(token)
                current_labels.append(label)

    # Don't forget the last sentence
    if current_tokens:
        all_tokens.append(current_tokens)
        all_labels.append(current_labels)

    return all_tokens, all_labels


def parse_labelstudio_json(filepath: str) -> tuple[list[list[str]], list[list[str]]]:
    """
    Parse Label Studio JSON export into token and label lists.

    Converts span-level annotations to BIO-tagged token sequences.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_tokens: list[list[str]] = []
    all_labels: list[list[str]] = []

    for task in data:
        text = task.get("data", {}).get("text", "")
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        # Get the first annotation result
        results = annotations[0].get("result", [])

        # Build character-level label map
        char_labels = ["O"] * len(text)
        for result in results:
            if result.get("type") != "labels":
                continue
            value = result.get("value", {})
            start = value.get("start", 0)
            end = value.get("end", 0)
            labels = value.get("labels", [])
            if not labels:
                continue
            label = labels[0]

            # Mark B- for first character, I- for rest
            for i in range(start, min(end, len(text))):
                if i == start:
                    char_labels[i] = f"B-{label}"
                else:
                    char_labels[i] = f"I-{label}"

        # Simple whitespace tokenization with label alignment
        tokens: list[str] = []
        labels: list[str] = []
        current_token = ""
        current_label = "O"

        for i, ch in enumerate(text):
            if ch in (" ", "\n", "\t"):
                if current_token:
                    tokens.append(current_token)
                    labels.append(current_label)
                    current_token = ""
                    current_label = "O"
            else:
                if not current_token:
                    current_label = char_labels[i]
                current_token += ch

        if current_token:
            tokens.append(current_token)
            labels.append(current_label)

        if tokens:
            all_tokens.append(tokens)
            all_labels.append(labels)

    return all_tokens, all_labels


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset(
    all_tokens: list[list[str]],
    all_labels: list[list[str]],
    tokenizer,
    label2id: dict[str, int],
    max_length: int = 512,
):
    """
    Convert tokenized sentences to HuggingFace Dataset with subword alignment.

    Handles the tokenizer's subword splitting by aligning labels to the
    first subword token of each word.
    """
    from datasets import Dataset

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for tokens, labels in zip(all_tokens, all_labels):
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a new word
                label_str = labels[word_idx] if word_idx < len(labels) else "O"
                aligned_labels.append(label2id.get(label_str, label2id.get("O", 0)))
            else:
                # Continuation subword — use I- label or -100
                label_str = labels[word_idx] if word_idx < len(labels) else "O"
                if label_str.startswith("B-"):
                    label_str = "I-" + label_str[2:]
                aligned_labels.append(label2id.get(label_str, label2id.get("O", 0)))
            previous_word_idx = word_idx

        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])
        labels_list.append(aligned_labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred, id2label: dict[int, str]):
    """Compute precision, recall, F1 using seqeval."""
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    import numpy as np

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Convert IDs back to label strings, ignoring -100
    true_labels = [
        [id2label[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_preds = [
        [id2label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    print("\n" + classification_report(true_labels, true_preds))

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


# ── Main fine-tuning function ─────────────────────────────────────────────────

def finetune(args):
    """Run the full fine-tuning pipeline."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    data_path = args.data
    print(f"\n  Loading annotations from: {data_path}")

    if data_path.endswith(".json"):
        all_tokens, all_labels = parse_labelstudio_json(data_path)
    else:
        all_tokens, all_labels = parse_conll_file(data_path)

    print(f"  Loaded {len(all_tokens)} annotated sentences")

    if len(all_tokens) < 5:
        print("  [ERROR] Need at least 5 annotated sentences. Aborting.")
        sys.exit(1)

    # ── Step 2: Build label vocabulary ────────────────────────────────────
    unique_labels = sorted(set(
        label for labels in all_labels for label in labels
    ))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"  Label vocabulary ({len(unique_labels)} labels):")
    for label in unique_labels:
        count = sum(1 for labels in all_labels for l in labels if l == label)
        print(f"    {label:<20} {count:>6} occurrences")

    # ── Step 3: Load tokenizer and model ──────────────────────────────────
    print(f"\n  Loading base model: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── Step 4: Prepare datasets ──────────────────────────────────────────
    print(f"  Preparing dataset (max_seq_length={args.max_seq_length})...")

    full_dataset = prepare_dataset(
        all_tokens, all_labels, tokenizer, label2id, args.max_seq_length
    )

    # Train/val split
    split = full_dataset.train_test_split(
        test_size=args.val_split, seed=args.seed
    )
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # ── Step 5: Training ──────────────────────────────────────────────────
    print(f"\n  Starting fine-tuning for {args.epochs} epochs...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    trainer.train()

    # ── Step 6: Save final model ──────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\n  Saving fine-tuned model to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save label mapping
    label_map_path = os.path.join(final_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\n  Fine-tuning complete!")
    print(f"  Model saved to: {final_dir}")
    print(f"\n  To use the fine-tuned model, update BIOBERT_NER_MODEL in")
    print(f"  backend/config.py to point to: {final_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BioBERT NER on annotated medical reports"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to annotated data (CoNLL 2003 or Label Studio JSON)"
    )
    parser.add_argument(
        "--output", default=DEFAULTS["output_dir"],
        dest="output_dir",
        help=f"Output directory for fine-tuned model (default: {DEFAULTS['output_dir']})"
    )
    parser.add_argument(
        "--base-model", default=DEFAULTS["base_model"],
        help=f"Base model to fine-tune (default: {DEFAULTS['base_model']})"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULTS["epochs"],
        help=f"Number of training epochs (default: {DEFAULTS['epochs']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULTS["batch_size"],
        help=f"Batch size (default: {DEFAULTS['batch_size']})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULTS["learning_rate"],
        help=f"Learning rate (default: {DEFAULTS['learning_rate']})"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=DEFAULTS["max_seq_length"],
        help=f"Max sequence length (default: {DEFAULTS['max_seq_length']})"
    )
    parser.add_argument(
        "--val-split", type=float, default=DEFAULTS["val_split"],
        help=f"Validation split ratio (default: {DEFAULTS['val_split']})"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULTS["seed"],
        help=f"Random seed (default: {DEFAULTS['seed']})"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int,
        default=DEFAULTS["early_stopping_patience"],
        help=f"Early stopping patience (default: {DEFAULTS['early_stopping_patience']})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    finetune(args)
