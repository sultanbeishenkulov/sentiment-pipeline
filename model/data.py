"""
Load the tweet_eval sentiment dataset and tokenize it for DistilBERT.

tweet_eval is a benchmark of Twitter classification tasks. The 'sentiment'
subset is 3-class (negative=0, neutral=1, positive=2) with ~60k labeled
tweets pre-split into train/val/test.

Why this dataset:
  - Domain match: short, informal social media text — same shape as Bluesky
  - Standard benchmark: published, reproducible, recruiter-recognizable
  - 3-class is the right granularity for a real dashboard
    (binary pos/neg makes everything look extreme)

Usage:
    from data import load_tokenized_datasets, LABEL2ID, ID2LABEL
    train_ds, val_ds, test_ds, tokenizer = load_tokenized_datasets()
"""
from __future__ import annotations

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128  # tweets are short; padding beyond this wastes compute

# tweet_eval label mapping (matches the dataset's int labels)
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_tokenized_datasets(
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
) -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    """Returns (tokenized DatasetDict with train/validation/test, tokenizer)."""
    raw = load_dataset("tweet_eval", "sentiment")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            # Pad dynamically per batch via DataCollatorWithPadding in train.py;
            # don't pad to max_length here — wastes memory on short tweets.
        )

    tokenized = raw.map(_tokenize, batched=True, remove_columns=["text"])
    # Trainer expects labels under the column name "labels"
    tokenized = tokenized.rename_column("label", "labels")

    return tokenized, tokenizer


if __name__ == "__main__":
    # Quick sanity check: load and print sizes + a sample row
    tokenized, tokenizer = load_tokenized_datasets()
    print("Splits:", {k: len(v) for k, v in tokenized.items()})
    print("Sample row keys:", list(tokenized["train"][0].keys()))
    print("Label distribution (train):")
    from collections import Counter
    print("  ", Counter(tokenized["train"]["labels"]))
    print("Label mapping:", ID2LABEL)
