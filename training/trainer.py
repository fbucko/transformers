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

# Logging function from the utils module
from utils.logger import log_metrics

def train_epoch(model, dataloader, optimizer, device, rank):
    """
    Runs a single training epoch.
    """
    model.train()
    total_loss = 0
    iterator = tqdm(dataloader, desc="Training", disable=(rank != 0))
    
    for batch in iterator:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if rank == 0:
            iterator.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    return acc, precision, recall, f1

def main_worker(gpu, ngpus_per_node, args, train_dataset, val_dataset, tokenizer, get_model_fn):
    """
    Worker process for distributed training.
    """
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)

    # Initialize distributed process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.DISTRIBUTED_PORT}',
        world_size=ngpus_per_node,
        rank=gpu
    )

    # Create directories for logs and saved models
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    saved_models_dir = os.path.join(os.getcwd(), "models", "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    # Create distributed samplers and data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=ngpus_per_node, rank=gpu, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=ngpus_per_node, rank=gpu, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, sampler=val_sampler)

    # Initialize model and wrap with DDP
    model = get_model_fn(args.PRETRAINED_MODEL_NAME)
    print(f" ===== Initializing {args.PRETRAINED_MODEL_NAME} ===== ")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    optimizer = optim.AdamW(model.parameters(), lr=args.LEARNING_RATE)

    # Only rank 0 will handle printing, logging, and saving the model
    run_id = None
    if gpu == 0:
        run_id = getattr(args, "run_id", None)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(args.EPOCHS):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, device, gpu)
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        if gpu == 0:
            print(f"\nEpoch {epoch+1}/{args.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation - Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, "
                  f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            # Prepare a separate log file for this run
            log_file = os.path.join(logs_dir, f"{args.TYPE}_{args.MODEL_SAVE_NAME}_training_logs_{run_id}.csv")
            log_metrics(run_id, epoch + 1, train_loss, val_acc, val_precision, val_recall, val_f1, log_file=log_file)

    # Save the trained model (only by the main process)
    if gpu == 0:
        model_save_path = os.path.join(saved_models_dir, f"{args.TYPE}_{args.MODEL_SAVE_NAME}_{run_id}.pt")
        # Save the underlying model state dict from DDP
        torch.save(model.module.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    dist.destroy_process_group()

def train(args, train_dataset, val_dataset, tokenizer, get_model_fn):
    """
    Spawns processes for distributed training.
    """
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(ngpus_per_node, args, train_dataset, val_dataset, tokenizer, get_model_fn)
    )
