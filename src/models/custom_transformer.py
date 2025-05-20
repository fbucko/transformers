"""
custom_transformer.py

Defines a custom Transformer-based neural network for sequence classification.
Includes:
- CustomTransformerClassifier: a lightweight Transformer encoder for embeddings.
- CustomHFWrapper: a wrapper providing HuggingFace-compatible interface and output.

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomTransformerClassifier(nn.Module):
    """
    A lightweight Transformer encoder for sequence classification tasks.
    """
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_classes: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos   = nn.Embedding(max_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        B, L = input_ids.size()
        pos_ids = (
            torch.arange(L, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, L)
        )
        x = self.embed(input_ids) + self.pos(pos_ids)
        x = self.encoder(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x), None  # attentions ignored

class CustomHFWrapper(nn.Module):
    """
    HuggingFace-compatible wrapper around the custom Transformer classifier.
    Returns SequenceClassifierOutput for integration with Trainer API.
    """
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_labels: int,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.core = CustomTransformerClassifier(
            vocab_size=vocab_size,
            max_length=max_length,
            num_classes=num_labels,
            pad_token_id=pad_token_id
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ) -> SequenceClassifierOutput:
        logits, _ = self.core(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
