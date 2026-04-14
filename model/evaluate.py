"""
Evaluate the fine-tuned model on the held-out test set.

Reports:
  - Per-class precision / recall / F1 (sklearn classification_report)
  - Confusion matrix (text + PNG)
  - Overall accuracy + macro/weighted F1

The test set is *only* used here, never during training, so these numbers
are an honest estimate of how the model will perform on unseen Bluesky data.

Run:
    python evaluate.py

Outputs:
    model/artifacts/eval_report.txt          # text classification report
    model/artifacts/confusion_matrix.png     # heatmap
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import ID2LABEL, load_tokenized_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
LABELS_ORDERED = [ID2LABEL[i] for i in range(len(ID2LABEL))]


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_confusion_matrix_png(cm: np.ndarray, path: Path) -> None:
    """Render the confusion matrix as a labeled heatmap."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive; safe for headless / scripts
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS_ORDERED)))
    ax.set_yticks(range(len(LABELS_ORDERED)))
    ax.set_xticklabels(LABELS_ORDERED)
    ax.set_yticklabels(LABELS_ORDERED)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — tweet_eval test set")

    # Annotate each cell with the count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    if not ARTIFACTS_DIR.exists() or not any(ARTIFACTS_DIR.iterdir()):
        raise SystemExit(
            f"No model found at {ARTIFACTS_DIR}. Run `python train.py` first."
        )

    device = _detect_device()
    logger.info("Evaluating on device: %s", device)

    logger.info("Loading tokenized test set...")
    tokenized, _ = load_tokenized_datasets()
    test_ds = tokenized["test"]
    logger.info("Test set size: %d", len(test_ds))

    logger.info("Loading fine-tuned model from %s", ARTIFACTS_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(ARTIFACTS_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(ARTIFACTS_DIR))

    # Use Trainer purely as a convenient batched-prediction harness
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/eval_tmp",
            per_device_eval_batch_size=64,
            report_to="none",
            use_cpu=(device == "cpu"),
        ),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logger.info("Running predictions...")
    output = trainer.predict(test_ds)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids

    # Headline metrics
    acc = accuracy_score(labels, preds)
    f1_m = f1_score(labels, preds, average="macro")
    f1_w = f1_score(labels, preds, average="weighted")

    report_text = classification_report(
        labels, preds,
        target_names=LABELS_ORDERED,
        digits=4,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS_ORDERED))))

    summary = (
        "Test-set evaluation\n"
        "===================\n"
        f"Accuracy:    {acc:.4f}\n"
        f"F1 (macro):  {f1_m:.4f}\n"
        f"F1 (weight): {f1_w:.4f}\n\n"
        "Per-class metrics\n"
        "-----------------\n"
        f"{report_text}\n"
        "Confusion matrix (rows=true, cols=pred)\n"
        "---------------------------------------\n"
        f"Labels: {LABELS_ORDERED}\n"
        f"{cm}\n"
    )

    print("\n" + summary)

    report_path = ARTIFACTS_DIR / "eval_report.txt"
    report_path.write_text(summary)
    logger.info("Wrote text report to %s", report_path)

    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    _save_confusion_matrix_png(cm, cm_path)
    logger.info("Wrote confusion matrix to %s", cm_path)


if __name__ == "__main__":
    main()
