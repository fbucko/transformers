"""
transformer_model_factory.py

Factory function for instantiating different types of sequence classification models.
Supports both HuggingFace's pretrained transformers and custom Transformer-based models.

Available model types:
- "standard": HuggingFace AutoModelForSequenceClassification
- "custom-char" / "custom-ngram" / "custom-pretrained-tokenizer": 
  lightweight Transformer wrapped in a HuggingFace-compatible interface

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""

import logging
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from .custom_transformer import CustomHFWrapper

logger = logging.getLogger(__name__)

def get_transformer_model(
    model_name: str,
    *,
    num_labels: int = 2,
    model_type: str = "standard",
    vocab_size: int = None,
    max_length: int = None,
    pad_token_id: int = 0,
) -> PreTrainedModel:
    """
    Factory that dispatches on model_type:
      - "standard": loads HF AutoModelForSequenceClassification
      - "custom-char" / "custom-ngram": returns our HF-compatible wrapper
    """
    model_type = model_type.lower()

    if model_type == "standard":
        logger.info(f"[Factory] Loading standard HF model '{model_name}'")
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    if model_type in ("custom-char", "custom-ngram","custom-pretrained-tokenizer"):
        if vocab_size is None or max_length is None:
            raise ValueError(
                "Custom model_type requires vocab_size and max_length arguments"
            )
        logger.info(f"[Factory] Loading custom model ({model_type})")
        return CustomHFWrapper(
            vocab_size=vocab_size,
            max_length=max_length,
            num_labels=num_labels,
            pad_token_id=pad_token_id
        )

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Use 'standard', 'custom-char', 'custom-ngram', or 'custom-pretrained-tokenizer'."
    )
