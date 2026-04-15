"""
DistilBERT sentiment inference.

Loads the fine-tuned model from Sprint 2 once at startup. Falls back to the
HuggingFace SST-2 base model if the fine-tuned artifact isn't there yet —
this way the consumer is runnable end-to-end even before Sprint 2 training
completes, and we can swap in the real model by dropping it into MODEL_PATH.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str   # uppercase: POSITIVE / NEGATIVE / NEUTRAL
    score: float # softmax prob of the predicted class, 0.0-1.0


class SentimentModel:
    def __init__(
        self,
        model_path: str | Path = "../model/artifacts",
        fallback_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str | None = None,
        max_length: int = 256,
    ):
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_path = Path(model_path)
        if model_path.exists() and any(model_path.iterdir()):
            source = str(model_path)
            logger.info("Loading fine-tuned model from %s", source)
        else:
            source = fallback_model
            logger.warning(
                "Fine-tuned model not found at %s — falling back to %s",
                model_path, fallback_model,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        scores, pred_ids = probs.max(dim=-1)

        return [
            SentimentResult(
                label=self.id2label[int(pid)].upper(),
                score=float(s),
            )
            for pid, s in zip(pred_ids.tolist(), scores.tolist())
        ]
