"""
Main distributed training script for Transformer-based text classification.

Supports:
- Multi-GPU distributed training (DDP via torch.multiprocessing)
- Custom or HuggingFace-compatible models and tokenizers
- Training and evaluation with early stopping and checkpointing
- Metric logging to CSV

Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 18.5.2024
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_scheduler, PreTrainedModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from src.utils.logger import log_metrics

# =========================================================
#                    Logging Init
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
#                  Train + Eval helpers
# =========================================================
def train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    rank: int,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (PreTrainedModel): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        device (torch.device): Device to use for computation.
        rank (int): Rank of the current GPU process (used for logging control).

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    loader = tqdm(dataloader, desc="Train", disable=(rank != 0))

    for batch in loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        if rank == 0:
            loader.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def evaluate(
    model: PreTrainedModel, dataloader: DataLoader, device: torch.device
) -> tuple[float, float, float, float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model (PreTrainedModel): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: Validation loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    total_loss = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            lbls = batch["label"].to(device)

            outputs = model(**inputs, labels=lbls)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(lbls.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return avg_loss, acc, prec, rec, f1


# =========================================================
#               Main worker (one per GPU)
# =========================================================
def main_worker(
    gpu: int,
    ngpus: int,
    args,
    train_dataset,
    val_dataset,
    get_model_fn: Callable[..., PreTrainedModel],
) -> None:
    """
    Main training logic executed per GPU in a distributed setup.

    Args:
        gpu (int): Local GPU index.
        ngpus (int): Total number of GPUs available.
        args: Argument namespace with training configuration.
        train_dataset: Training dataset instance.
        val_dataset: Validation dataset instance.
        get_model_fn (Callable): Function that returns a model instance.
    """
    # 1. Initialise GPU + process group 
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{args.DISTRIBUTED_PORT}",
        world_size=ngpus,
        rank=gpu,
    )

    # 2. Create directory structure 
    #    logs/<data_type>/... and models/<data_type>/...
    project_root = Path(os.getcwd())
    log_root     = project_root / "logs"   / args.DATA_TYPE
    ckpt_root    = project_root / "models" / args.DATA_TYPE
    log_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # 2.1. Set consistent run-id everywhere
    stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch_name  = args.PRETRAINED_MODEL_NAME.split("/")[-1]
    file_stem  = f"{args.TYPE}_{arch_name}_{stamp}"
    run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file   = log_root  / f"{file_stem}.csv"
    best_ckpt  = ckpt_root / f"{file_stem}_BEST.pt"
    final_ckpt = ckpt_root / f"{file_stem}_FINAL.pt"

    # 3. Initialization of the data loaders 
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=ngpus, rank=gpu, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=ngpus, rank=gpu, shuffle=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.BATCH_SIZE, sampler=val_sampler
    )

    # 4. Initialization of the Model, Optimiser and Scheduler 
    model = get_model_fn(
        args.PRETRAINED_MODEL_NAME, num_labels=args.NUM_LABELS
    ).to(device)
    model = DDP(model, device_ids=[gpu], output_device=gpu)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if name.endswith(("bias", "LayerNorm.weight")) else decay).append(
            param
        )
    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": args.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.LEARNING_RATE,
    )

    total_steps = len(train_loader) * args.EPOCHS
    warmup_steps = int(total_steps * args.WARMUP_RATIO)
    scheduler = get_scheduler(
        name=args.LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 5. Training loop
    best_metric = float("inf") if args.CKPT_MONITOR == "loss" else -float("inf")
    epochs_no_improve = 0
    stop_flag = torch.tensor([0], device=device)

    for epoch in range(1, args.EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, gpu)
        val_loss, acc, prec, rec, f1 = evaluate(model, val_loader, device)

        if gpu == 0:
            summary = (
                f"[GPU0]  epoch {epoch:>2d}/{args.EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}"
            )
            print(summary, flush=True)
            logger.info(summary) 
            
            # 5.1. Logging metrics
            log_metrics(
                run_id=run_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=acc,
                val_precision=prec,
                val_recall=rec,
                val_f1=f1,
                log_file=str(log_file),
            )

            # 5.2 Early stopping/Checkpointing
            metric = val_loss if args.CKPT_MONITOR == "loss" else f1
            improved = (
                metric < best_metric
                if args.CKPT_MONITOR == "loss"
                else metric > best_metric
            )
            if improved:
                best_metric = metric
                epochs_no_improve = 0
                torch.save(model.module.state_dict(), best_ckpt)
                logger.info(f"[GPU0]  Epoch {epoch:>2d}/{args.EPOCHS}: new BEST -> {best_ckpt}")
            else:
                epochs_no_improve += 1
                logger.info(f"[GPU0]  Epoch {epoch:>2d}/{args.EPOCHS}: no improvement ({epochs_no_improve})")

            if epochs_no_improve >= args.PATIENCE:
                logger.info("[GPU0] Early stopping triggered")
                stop_flag[0] = 1

        # 6. Broadcast stop signal to all ranks
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    # 7.  Final save
    if gpu == 0:
        torch.save(model.module.state_dict(), final_ckpt)
        logger.info(f"[GPU0] Saved FINAL model to {final_ckpt}")

    dist.destroy_process_group()



# =========================================================
#           Entry point (spawns 1 worker per GPU) 
# =========================================================
def train(
    args,
    train_dataset,
    val_dataset,
    get_model_fn: Callable[..., PreTrainedModel],
) -> None:
    """
    Launches distributed training using torch.multiprocessing.

    Args:
        args: Argument namespace with training configuration.
        train_dataset: Training dataset instance.
        val_dataset: Validation dataset instance.
        get_model_fn (Callable): Function that returns a model instance.
    """
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise EnvironmentError(
            "No CUDA devices found. This training script requires at least one GPU.\n"
            "Please ensure a compatible NVIDIA GPU and CUDA toolkit are available."
        )
    mp.spawn(
        main_worker,
        nprocs=ngpus,
        args=(ngpus, args, train_dataset, val_dataset, get_model_fn),
    )
