# Transformer-Based Malicious Domain Detection

This repository contains tools and code for building transformer-based models to detect malicious domains. It includes scripts for preprocessing, training, evaluation, and hyperparameter optimization. The pipeline is built on Python, PyTorch, and Hugging Face Transformers, and is designed for use in Conda environments.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Running Experiments](#running-experiments)
- [Pretrained Model Training](#pretrained-model-training)
- [Custom Architecture Training](#custom-architecture-training)
- [Hyperparameter Search](#hyperparameter-search)
- [Dataset Overview](#dataset-overview)

## Project Structure

```
.
├── datasets/            # Benign, malware, phishing, DGA datasets (CSV/JSON)
├── transformers/        # Config files, training/evaluation notebooks, logs, models
├── preprocessing/       # Data cleaning and formatting scripts
├── src/                 # Core modules: models, tokenizers, training, utilities
├── models/              # Trained model checkpoints
├── logs/                # Training logs and performance outputs
├── envs/                # Conda environment definitions
└── README.md            # Project documentation
```

## Installation

To run the project, a CUDA-enabled GPU is required. All setup steps assume the use of Conda for environment and dependency management.

### Step 1: Create Environments

The repository provides two environments:

- `gpu` for model training
- `hpsearch` for hyperparameter optimization

Create them using:

```bash
conda env create -f envs/gpu.yaml
conda env create -f envs/hpsearch.yaml
```

### Step 2: Activate the Environment

```bash
conda activate gpu        # For training
conda activate hpsearch   # For hyperparameter search
```

### Step 3: Verify Installation

```bash
conda list
```

## Running Experiments

### Environment-Specific Scripts

| Environment | Purpose                 | Script                                               |
|-------------|-------------------------|------------------------------------------------------|
| gpu         | Model training          | `experiments/custom_architecture_experiment.py`      |
| gpu         | Pretrained experiments | `experiments/pretrained_experiment.py`               |
| hpsearch    | Hyperparameter tuning   | `experiments/hyperparameter_search.py`               |

## Pretrained Model Training

To run experiments using pretrained models (e.g., BERT, DistilBERT, Electra):

```bash
python -m experiments.run_experiment
```

To override configuration values via command line using Hydra:

```bash
python -m experiments.run_experiment train.batch_size=512 model=google_electra_base
```

To run a specific experiment configuration:

```bash
python -m experiments.run_experiment -m experiments=rdap_dns_geo
```

## Custom Architecture Training

To train a custom transformer model:

```bash
python -m experiments.custom_architecture_experiment
```

Configuration file: `configs/custom_config.yaml`

You can customize tokenizer and architecture via command-line arguments or by editing the config.

## Hyperparameter Search

Run hyperparameter tuning with:

```bash
python -m experiments.hyperparameter_search
```

Before running, update the script to set:

1. **Dataset Path**:

```python
data_path = "datasets/dga/dga_preprocessed.csv"
# data_path = "datasets/malware/malware_preprocessed.csv"
# data_path = "datasets/phishing/phishing_preprocessed.csv"
```

2. **Model Architecture**:

```python
MODEL_NAME = "prajjwal1/bert-medium"
# MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "prajjwal1/bert-tiny"
```

## Dataset Overview

The `datasets/` directory contains input data categorized as follows:

- `benign/`: Clean traffic domain data in CSV and JSON formats
- `malware/`: Domains known to be associated with malware
- `phishing/`: Domains flagged for phishing activity
- `dga/`: Algorithmically generated domain names (DGA)

Each subdirectory may contain raw, anonymized, and preprocessed formats to support various stages of experimentation.

---