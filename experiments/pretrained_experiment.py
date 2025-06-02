#!/usr/bin/env python
"""
run_experiment.py

Main launcher for training Transformer models using Hydra-based configuration.

Supports:
- Single training jobs using a base YAML config
- Multi-job batches by listing `experiments.jobs` in the config

Each job can override any part of the base configuration (model, data, training).

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""
import sys
from pathlib import Path
from typing import Dict, List
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.data.dataset_loader import DatasetLoader
from src.data.datasets import DomainDataset, RDAPDataset, DNSDataset, GEODataset
from src.models.transformer_model_factory import get_transformer_model
from src.training.trainer import train

from argparse import Namespace

def cfg_to_namespace(c: DictConfig) -> Namespace:
    """
    Flatten the DictConfig into an argparse-like Namespace whose keys
    match what trainer.py expects (upper-case).
    """
    return Namespace(
        DATA_TYPE        = c.data.type,
        TYPE             = c.task.task,
        DISTRIBUTED_PORT = c.distributed_port,

        PRETRAINED_MODEL_NAME = c.model.name,
        NUM_LABELS       = c.model.num_labels,

        BATCH_SIZE       = c.train.batch_size,
        EPOCHS           = c.train.epochs,
        LEARNING_RATE    = c.train.learning_rate,
        WEIGHT_DECAY     = c.train.weight_decay,
        LR_SCHEDULER     = c.train.lr_scheduler,
        WARMUP_RATIO     = c.train.warmup_ratio,
        PATIENCE         = c.train.patience,
        CKPT_MONITOR     = c.train.ckpt_monitor,
    )

# =======================================================================
# 1. Directory where Hydra should look for YAML configs
#    (relative to this file -> script must be launched from project root)
# =======================================================================
CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").as_posix()

# =======================================================================
# 2. Helper function for executing 1 training job                                        
# =======================================================================
def _run_single_job(job_cfg: DictConfig) -> None:
    """
    Runs a single training job using the provided configuration.
    """
    # 1. Data loading
    data_path = to_absolute_path(job_cfg.data.path)
    loader = DatasetLoader(data_path, job_cfg.data.type)
    df = loader.load()

    train_df, val_df = train_test_split(
        df,
        test_size=job_cfg.data.val_split,
        random_state=job_cfg.seed,
        stratify=df.get("label") if job_cfg.data.type == "domain" else None,
    )

    DatasetCls = {
        "domain": DomainDataset,
        "rdap":   RDAPDataset,
        "dns":    DNSDataset,
        "geo":   GEODataset,
    }[job_cfg.data.type]    
    
    # 2. Setting the max_length from the task.yaml config
    safe_max = job_cfg.task.max_lengths[job_cfg.data.type]
    print(f"→ using max_length={safe_max} for {job_cfg.task.task}/{job_cfg.data.type}")
    job_cfg.data.max_length = safe_max

     # 3. Initialization of tokenizer and Datasets 
    tokenizer = AutoTokenizer.from_pretrained(job_cfg.model.name)

    train_ds = DatasetCls(train_df, tokenizer=tokenizer,
                      max_length=job_cfg.data.max_length)
    val_ds   = DatasetCls(val_df,   tokenizer=tokenizer,
                      max_length=job_cfg.data.max_length)

    # 4. Run training
    print(
        f"\n[run_experiment] data={job_cfg.data.type} - task={job_cfg.task}"
        f"Model={job_cfg.model.name} - cwd={Path.cwd().as_posix()}"
    )
    args_ns = cfg_to_namespace(job_cfg)
    print(args_ns.BATCH_SIZE)
    train(
        args=args_ns,
        train_dataset=train_ds,
        val_dataset=val_ds,
        get_model_fn=get_transformer_model,
    )


# ================================================================
#                           Main 
# ================================================================
@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="pretrained_config"
)
def main(cfg: DictConfig) -> None:
    """
    Entry point for training: launches either a single job or multiple jobs
    based on whether `experiments.jobs` is present in the config.
    """
    # Show the fully-resolved config for debugging
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # 1. Try to grab a list of jobs from the 'experiments' group
    jobs = getattr(cfg.experiments, "jobs", None) if cfg.get("experiments") else None

    # 2. If a list of jobs was returned, run in multi-job mode
    if jobs:
        print(f"→ Multi-job mode: {len(jobs)} runs\n")
        for idx, overrides in enumerate(jobs, start=1):
            print(f"==== RUN {idx}/{len(jobs)} ====")
            # Merge per-job overrides into the base config
            job_cfg = OmegaConf.merge(cfg, overrides)
            print(
                f"Running: data={job_cfg.data.type}, "
                f"task={job_cfg.task}, "
                f"model={job_cfg.model.name}"
            )
            _run_single_job(job_cfg)

    # 3. Otherwise, run single-job mode
    else:
        print("→ Single-job mode")
        _run_single_job(cfg)


if __name__ == "__main__":
    main()
