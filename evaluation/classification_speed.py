import time
import torch
import pandas as pd
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# 1. CONFIGURATION: base model for tokenizer/config, plus your .pt checkpoint
CONFIG = {
    "Domain-DGA-BERT-medium": {
        "base_model": "prajjwal1/bert-medium",
        "ckpt_path": "../models/domain/BEST_dga_prajjwal1-bert-medium_20250503_145342.pt",
        "csv_path": "../../datasets/dga/dga_preprocessed.csv",
        "col": "domain_name",
    },
    "Domain-Malware-ELECTRA": {
        "base_model": "google/electra-base-discriminator",
        "ckpt_path": "../models/domain/BEST_malware_google-electra-base-discriminator_20250504_001819.pt",
        "csv_path": "../../datasets/malware/malware_preprocessed.csv",
        "col": "domain_name",
    },
    "Domain-Phishing-Distil": {
        "base_model": "distilbert-base-uncased",
        "ckpt_path": "../models/domain/BEST_phish_distilbert-base-uncased_20250504_014049.pt",
        "csv_path": "../../datasets/phishing/phishing_preprocessed.csv",
        "col": "domain_name",
    },
    "RDAP-Malware-ELECTRA": {
        "base_model": "google/electra-base-discriminator",
        "ckpt_path": "../models/rdap/malware_electra-base-discriminator_20250512_174759_BEST.pt",
        "csv_path": "../../datasets/malware/rdap_malware_preprocessed.csv",
        "col": "input_string",
    },
    "RDAP-Phishing-Distil": {
        "base_model": "distilbert-base-uncased",
        "ckpt_path": "../models/rdap/phishing_distilbert-base-uncased_20250512_201957_BEST.pt",
        "csv_path": "../../datasets/phishing/rdap_phishing_preprocessed.csv",
        "col": "input_string",
    },
    "DNS-Malware-ELECTRA": {
        "base_model": "google/electra-base-discriminator",
        "ckpt_path": "../models/dns/malware_electra-base-discriminator_20250512_220139_BEST.pt",
        "csv_path": "../../datasets/malware/dns_malware_preprocessed.csv",
        "col": "input_string",
    },
    "DNS-Phishing-Distil": {
        "base_model": "distilbert-base-uncased",
        "ckpt_path": "../models/dns/phishing_distilbert-base-uncased_20250512_233838_BEST.pt",
        "csv_path": "../../datasets/phishing/dns_phishing_preprocessed.csv",
        "col": "input_string",
    },
    "Geo-Malware-ELECTRA": {
        "base_model": "google/electra-base-discriminator",
        "ckpt_path": "../models/geo/malware_electra-base-discriminator_20250513_010704_BEST.pt",
        "csv_path": "../../datasets/malware/geo_malware_preprocessed.csv",
        "col": "input_string",
    },
    "Geo-Phishing-Distil": {
        "base_model": "distilbert-base-uncased",
        "ckpt_path": "../models/geo/phishing_distilbert-base-uncased_20250513_013233_BEST.pt",
        "csv_path": "../../datasets/phishing/geo_phishing_preprocessed.csv",
        "col": "input_string",
    },
}

# 2. Device selection (GPU if available)
device = 0 if torch.cuda.is_available() else -1
print(f"Running inference on {'GPU' if device>=0 else 'CPU'}\n")

# 3. Load pipelines, unwrapping any saved state-dict
pipes = {}
for name, conf in CONFIG.items():
    print(f"Loading {name} …")
    # 3.1 tokenizer + config from the base pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(conf["base_model"])
    config    = AutoConfig.from_pretrained(conf["base_model"])
    # 3.2 instantiate fresh model
    model     = AutoModelForSequenceClassification.from_config(config)

    # 3.3 load your .pt checkpoint
    raw = torch.load(conf["ckpt_path"], map_location="cpu")
    # unwrap common wrappers
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw

    # strip any 'model.' or 'module.' prefixes
    cleaned = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}

    # load into model with strict=False to allow missing/backbone mismatches
    model.load_state_dict(cleaned, strict=False)

    # 3.4 move to device & set eval mode
    model.to(f"cuda:{device}" if device>=0 else "cpu").eval()

    # 3.5 wrap in HF pipeline
    pipes[name] = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=False,
    )
print()

# 4. Sample test inputs (500 examples per model)
SAMPLE_SIZE = 500
test_samples = {}
for name, conf in CONFIG.items():
    df = pd.read_csv(conf["csv_path"], usecols=[conf["col"]])
    test_samples[name] = (
        df[conf["col"]]
        .astype(str)
        .sample(SAMPLE_SIZE, random_state=42)
        .tolist()
    )

# 5. Benchmark: warm-up + timed batch inference
results = []
for name, pipe in pipes.items():
    samples = test_samples[name]
    _ = pipe(samples[:10], batch_size=16)         # warm-up
    t0 = time.perf_counter()
    _ = pipe(samples, batch_size=16)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / SAMPLE_SIZE * 1000
    results.append({"model": name, "avg_latency_ms": avg_ms})
    print(f"{name}: {avg_ms:.2f} ms/sample")

# 6. Summary
print("\nAverage latency per sample (ms):")
print(pd.DataFrame(results).sort_values("avg_latency_ms").to_string(index=False))
