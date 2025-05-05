# training/trainer.py
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
from transformers import get_scheduler
from utils.logger import log_metrics


# ───────────────────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device, rank):  
    model.train()
    total_loss = 0.0
    iterator = tqdm(dataloader, desc="Training", disable=(rank != 0))

    for batch in iterator:
        optimizer.zero_grad()

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if rank == 0:
            iterator.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


# ───────────────────────────────────────────────────────────────────────────────
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss   = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_loss / len(dataloader)
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )

    return avg_val_loss, acc, precision, recall, f1


# ───────────────────────────────────────────────────────────────────────────────
def main_worker(gpu, ngpus_per_node, args, train_dataset, val_dataset,
                tokenizer, get_model_fn):

    # Basic DDP setup -----------------------------------------------------------
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.DISTRIBUTED_PORT}',
        world_size=ngpus_per_node,
        rank=gpu
    )

    # I/O paths -----------------------------------------------------------------
    logs_dir          = os.path.join(os.getcwd(), "logs")
    saved_models_dir  = os.path.join(os.getcwd(), "models", "saved_models")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    # Data loaders --------------------------------------------------------------
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=ngpus_per_node,
                                       rank=gpu, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,
                                       num_replicas=ngpus_per_node,
                                       rank=gpu, shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.BATCH_SIZE,
                              sampler=train_sampler)
    val_loader   = DataLoader(val_dataset,
                              batch_size=args.BATCH_SIZE,
                              sampler=val_sampler)

    # Model ---------------------------------------------------------------------
    model = get_model_fn(args.PRETRAINED_MODEL_NAME)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], output_device=gpu
    )

    # Optimizer with selective weight decay ------------------------------------
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(("bias", "LayerNorm.weight")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=args.LEARNING_RATE
    )

    # Scheduler with warm‑up ----------------------------------------------------
    total_steps  = len(train_loader) * args.EPOCHS
    warmup_steps = int(total_steps * args.WARMUP_RATIO)

    scheduler = get_scheduler(
        name=args.LR_SCHEDULER,          
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Book‑keeping --------------------------------------------------------------
    run_id = None
    if gpu == 0:
        run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")

    best_val_score    = float("inf") if args.CKPT_MONITOR == "loss" else -1.0
    epochs_no_improve = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(args.EPOCHS):
        train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, gpu)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, device
        )

        # ----- Only rank 0 prints / logs / checkpoints -------------------------
        if gpu == 0:
            print(f"\nEpoch {epoch+1}/{args.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f" Val  Loss: {val_loss:.4f}")
            print(f"Validation - Acc: {val_acc:.4f}, "
                  f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

            log_metrics(
                run_id, epoch+1, train_loss,
                val_acc, val_prec, val_rec, val_f1,
                val_loss=val_loss,
                log_file=os.path.join(
                    logs_dir,
                    f"{args.TYPE}_{args.MODEL_SAVE_NAME}_training_logs_{run_id}.csv"
                )
            )

            # Early‑stopping / checkpoint logic ---------------------------------
            monitor_val = val_loss if args.CKPT_MONITOR == "loss" else val_f1

            improved = (
                (args.CKPT_MONITOR == "loss" and monitor_val < best_val_score) or
                (args.CKPT_MONITOR != "loss" and monitor_val > best_val_score)
            )

            if improved:
                best_val_score    = monitor_val
                epochs_no_improve = 0
                best_path = os.path.join(
                    saved_models_dir,
                    f"BEST_{args.TYPE}_{args.MODEL_SAVE_NAME}_{run_id}.pt"
                )
                torch.save(model.module.state_dict(), best_path)
                print(f"✓ New best model saved to {best_path}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")

            stop_flag = 1 if epochs_no_improve >= args.PATIENCE else 0
        else:
            stop_flag = 0   # still define for broadcast

        # Sync the stop decision across ranks -----------------------------------
        stop_tensor = torch.tensor([stop_flag], device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            if gpu == 0:
                print(f"⏹ Early stopping after {epoch+1} epochs.")
            break

    # Final checkpoint ----------------------------------------------------------
    if gpu == 0:
        final_path = os.path.join(
            saved_models_dir,
            f"{args.TYPE}_{args.MODEL_SAVE_NAME}_{run_id}.pt"
        )
        torch.save(model.module.state_dict(), final_path)
        print(f"Final model saved to {final_path}")

    dist.destroy_process_group()


# ───────────────────────────────────────────────────────────────────────────────
def train(args, train_dataset, val_dataset, tokenizer, get_model_fn):
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(ngpus_per_node, args, train_dataset, val_dataset,
              tokenizer, get_model_fn)
    )
