"""
custom_tokenizer.py

Implements a simple tokenizer for character or n-gram level tokenization,
suitable for training custom Transformer models without relying on pretrained tokenizers.

Features:
- Character-level or n-gram-based tokenization
- Vocabulary building from raw text
- HuggingFace-compatible __call__ interface with input IDs and attention masks

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""

from typing import List, Dict
import torch

class CustomTokenizer:
    """
    A simple tokenizer for char-level or n-gram-level tokenization.

    Args:
        strategy (str): "char" or "ngram"
        max_length (int): Max sequence length
        ngram_size (int): Size of n-grams (used if strategy != "char")
    """
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, strategy: str="char", max_length: int=64, ngram_size: int=3):
        self.strategy   = strategy
        self.max_length = max_length
        self.ngram_size = ngram_size
        self.vocab: Dict[str,int] = {self.PAD:0, self.UNK:1}

    def build_vocab(self, texts: List[str]) -> None:
        """
        Builds vocabulary from a list of input texts based on the selected strategy.
        """
        units = set()
        if self.strategy == "char":
            for t in texts:
                units.update(t)
        else:
            for t in texts:
                for i in range(len(t)-self.ngram_size+1):
                    units.add(t[i:i+self.ngram_size])
        for u in sorted(units):
            self.vocab.setdefault(u, len(self.vocab))

    def __call__(self, text, *, max_length=None, padding="max_length",
                 truncation=True, return_tensors="pt"):
        single = isinstance(text, str)
        texts = [text] if single else text
        encs = [
            self._encode_one(t, max_length or self.max_length, padding, truncation)
            for t in texts
        ]
        ids  = torch.tensor([e["input_ids"] for e in encs])
        mask = torch.tensor([e["attention_mask"] for e in encs])
        out = {"input_ids": ids, "attention_mask": mask}
        return {k:v.squeeze(0) for k,v in out.items()} if single else out

    def _encode_one(self, text, max_length, padding, truncation):
        if self.strategy == "char":
            toks = list(text)
        else:
            toks = [
                text[i:i+self.ngram_size]
                for i in range(len(text)-self.ngram_size+1)
            ]
        ids = [self.vocab.get(t, self.vocab[self.UNK]) for t in toks]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        if padding == "max_length" and len(ids) < max_length:
            ids += [self.vocab[self.PAD]] * (max_length - len(ids))
        mask = [1 if i != self.vocab[self.PAD] else 0 for i in ids]
        return {"input_ids": ids, "attention_mask": mask}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD]
