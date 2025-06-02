#!/usr/bin/env python
"""
custom_architecture_experiment.py

Launches training using a custom Transformer architecture and tokenizer,
configured via Hydra.

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

# 1.Set the working directory to the root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig
from argparse import Namespace
from functools import partial
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader              import DatasetLoader
from src.data.datasets                    import DomainDataset, RDAPDataset, DNSDataset, GEODataset
from src.tokenizer.tokenizer_factory     import get_tokenizer
from src.models.transformer_model_factory import get_transformer_model
from src.training.trainer                 import train  

# 2. Point Hydra at the root configs 
@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="custom_config"
)
def main(cfg: DictConfig):
    # 1. Load data
    loader = DatasetLoader(cfg.data.path, cfg.data.type)
    df     = loader.load()
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.data.val_split,
        random_state=cfg.seed,
        stratify=df.get("label") if cfg.data.type == "domain" else None,
    )
    # 2. Tokenizer
    texts = train_df.iloc[:, 0].tolist()
    tokenizer = get_tokenizer(
        strategy              = cfg.model.tokenizer_strategy,
        pretrained_model_name = cfg.model.name,
        max_length            = cfg.data.max_length,
        ngram_size            = cfg.model.ngram_size,
        build_vocab_texts     = texts,
    )
    # 3. Initialize datasets
    DatasetCls = {
        "domain": DomainDataset,
        "rdap":   RDAPDataset,
        "dns":    DNSDataset,
        "geo":    GEODataset,
    }[cfg.data.type]
    train_ds = DatasetCls(train_df, tokenizer=tokenizer, max_length=cfg.data.max_length)
    val_ds   = DatasetCls(val_df,   tokenizer=tokenizer, max_length=cfg.data.max_length)
    
    # 4. Wrap of the model factory for compatibility with DDP
    model_builder = partial(
        get_transformer_model,
        num_labels   = cfg.model.num_labels,
        model_type   = cfg.model.model_type,
        vocab_size   = tokenizer.vocab_size,
        max_length   = cfg.data.max_length,
        pad_token_id = tokenizer.pad_token_id,
    )
    
    # 5. Namespace mapping for compatibility with DDP trainer
    args_ns = Namespace(
        DATA_TYPE             = cfg.data.type,
        TYPE                  = cfg.task.task,
        DISTRIBUTED_PORT      = cfg.distributed_port,
        PRETRAINED_MODEL_NAME = cfg.model.name,
        NUM_LABELS            = cfg.model.num_labels,
        BATCH_SIZE            = cfg.train.batch_size,
        EPOCHS                = cfg.train.epochs,
        LEARNING_RATE         = cfg.train.learning_rate,
        WEIGHT_DECAY          = cfg.train.weight_decay,
        LR_SCHEDULER          = cfg.train.lr_scheduler,
        WARMUP_RATIO          = cfg.train.warmup_ratio,
        PATIENCE              = cfg.train.patience,
        CKPT_MONITOR          = cfg.train.ckpt_monitor,
    )
    
    # 6. Run the training
    train(
        args          = args_ns,
        train_dataset = train_ds,
        val_dataset   = val_ds,
        get_model_fn  = model_builder,
    )
if __name__ == "__main__":
    main()
