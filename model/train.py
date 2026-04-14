"""
Fine-tune DistilBERT on tweet_eval sentiment.

Design choices:
  - Trainer API: standard, well-tested, less code to maintain than a custom loop
  - Early stopping on validation macro-F1 (not loss): F1 is what we actually
    care about for this task, especially since classes are imbalanced
  - Save best checkpoint, not last: the last epoch isn't always the best
  - Auto-detect device: MPS (Apple Silicon) > CPU; CUDA if a real GPU exists
  - Fixed seed for reproducibility

Run:
    python train.py

Outputs:
    model/artifacts/         # final saved model + tokenizer
    model/training_logs/     # per-step training metrics
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import ID2LABEL, LABEL2ID, MODEL_NAME, load_tokenized_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---- config ---------------------------------------------------------------

SEED = 42
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
MAX_EPOCHS = 5
WARMUP_RATIO = 0.1   # warm up over first 10% of training steps
WEIGHT_DECAY = 0.01

OUTPUT_DIR = Path(__file__).parent / "artifacts"
LOGS_DIR = Path(__file__).parent / "training_logs"


# ---- device detection -----------------------------------------------------

def _detect_device() -> str:
    """Pick the best available device. Allow override via FORCE_CPU=1 in case
    MPS misbehaves (which it occasionally does on certain HF ops)."""
    if os.getenv("FORCE_CPU") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---- metrics --------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """Called by Trainer after each eval. Returns metrics dict."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ---- main -----------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    device = _detect_device()
    logger.info("Training on device: %s", device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and tokenizing tweet_eval/sentiment...")
    tokenized, tokenizer = load_tokenized_datasets()
    logger.info("Splits: %s", {k: len(v) for k, v in tokenized.items()})

    logger.info("Loading DistilBERT base for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Dynamic padding per batch — much faster than padding to max_length globally
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # MPS doesn't support fp16 reliably as of mid-2025; only enable on CUDA
    use_fp16 = device == "cuda"

    args = TrainingArguments(
        output_dir=str(LOGS_DIR),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # eval has no backprop, can fit more
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,           # keep only the 2 most recent checkpoints
        logging_steps=50,
        report_to="none",             # skip wandb/tensorboard for now
        fp16=use_fp16,
        seed=SEED,
        # Force device explicitly — Trainer auto-detects but being explicit
        # avoids surprises if env vars override.
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model to %s", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # One final eval on validation set for the log
    logger.info("Final validation metrics:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
