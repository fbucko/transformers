"""
tokenizer_factory.py

Provides a factory function to return a tokenizer instance based on a specified strategy.

Supported strategies:
- "char" / "ngram": uses a custom tokenizer with optional vocabulary building
- "pretrained-tokenizer": uses a HuggingFace tokenizer (default: distilbert-base-uncased)
- any other string: interpreted as a HuggingFace model name

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""
from transformers import AutoTokenizer
from .custom_tokenizer import CustomTokenizer

def get_tokenizer(
    strategy: str,
    pretrained_model_name: str,
    max_length: int,
    ngram_size: int,
    build_vocab_texts=None
):
    if strategy in ("char", "ngram"):
        tok = CustomTokenizer(
            strategy=strategy,
            max_length=max_length,
            ngram_size=ngram_size
        )
        if build_vocab_texts is None:
            raise ValueError("Must pass build_vocab_texts for custom tokenizer")
        tok.build_vocab(build_vocab_texts)
        return tok

    if strategy == "pretrained-tokenizer":
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # fallback: Hugging Face tokenizer
    return AutoTokenizer.from_pretrained(pretrained_model_name)
