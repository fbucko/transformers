"""
Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: May 21, 2025
Description:
    Computes and displays the sizes of model checkpoint files defined in CONFIG.
    Reports each file's size in megabytes or marks as "N/A" if inaccessible.
"""
import os
import pandas as pd

# 1. CONFIGURATION: same model→.pt paths as before
CONFIG = {
    "Domain-DGA-BERT-medium": {
        "ckpt_path": "../models/domain/BEST_dga_prajjwal1-bert-medium_20250503_145342.pt",
    },
    "Domain-Malware-ELECTRA": {
        "ckpt_path": "../models/domain/BEST_malware_google-electra-base-discriminator_20250504_001819.pt",
    },
    "Domain-Phishing-Distil": {
        "ckpt_path": "../models/domain/BEST_phish_distilbert-base-uncased_20250504_014049.pt",
    },
    "RDAP-Malware-ELECTRA": {
        "ckpt_path": "../models/rdap/malware_electra-base-discriminator_20250512_174759_BEST.pt",
    },
    "RDAP-Phishing-Distil": {
        "ckpt_path": "../models/rdap/phishing_distilbert-base-uncased_20250512_201957_BEST.pt",
    },
    "DNS-Malware-ELECTRA": {
        "ckpt_path": "../models/dns/malware_electra-base-discriminator_20250512_220139_BEST.pt",
    },
    "DNS-Phishing-Distil": {
        "ckpt_path": "../models/dns/phishing_distilbert-base-uncased_20250512_233838_BEST.pt",
    },
    "Geo-Malware-ELECTRA": {
        "ckpt_path": "../models/geo/malware_electra-base-discriminator_20250513_010704_BEST.pt",
    },
    "Geo-Phishing-Distil": {
        "ckpt_path": "../models/geo/phishing_distilbert-base-uncased_20250513_013233_BEST.pt",
    },
}

# 2. Gather sizes
rows = []
for name, conf in CONFIG.items():
    path = conf["ckpt_path"]
    try:
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024**2)
    except OSError:
        size_mb = None
    rows.append({
        "model": name,
        "path": path,
        "size_mb": size_mb,
    })

# 3. Build DataFrame and pretty-print
df = pd.DataFrame(rows)
df["size_mb"] = df["size_mb"].map(lambda x: f"{x:.2f}" if x is not None else "N/A")
print(df[["model", "size_mb"]].to_string(index=False))
