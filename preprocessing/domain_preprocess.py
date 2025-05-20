"""
Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: 19.5.2025
Description:
    Samples and balances phishing (and optionally malware) vs. benign domain datasets.
    Reads source CSVs, creates labeled subsets, shuffles, and writes out a preprocessed CSV.
"""

import pandas as pd
from pathlib import Path

def find_project_root() -> Path:
    """
    Walk up from this script's directory until a 'datasets' folder is found.
    Returns the project root containing 'datasets', or raises if not found.
    """
    current = Path(__file__).resolve().parent
    while True:
        if (current / "datasets").exists():
            return current
        if current.parent == current:
            raise RuntimeError("Could not find 'datasets' directory in any parent path.")
        current = current.parent

def main():
    # Locate project root and datasets directory
    project_root = find_project_root()
    dataset_dir = project_root / "datasets"

    # Load raw datasets
    phishing_df = pd.read_csv(dataset_dir / "phishing" / "phishing_strict_2024.csv")   # columns: domain_name, URL
    # -- To process malware instead of phishing, uncomment:
    # malware_df = pd.read_csv(dataset_dir / "malware_strict_2024.csv")

    benign1_df = pd.read_csv(dataset_dir / "benign" / "benign_2312_anonymized.csv")    # column: domain_name
    benign2_df = pd.read_csv(dataset_dir / "benign" / "umbrella_benign_FINISHED.csv")  # column: domain_name

    # Sample 100,000 phishing domains
    phishing_sample = (
        phishing_df[["domain_name"]]
        .sample(n=100_000, random_state=42)
        .assign(label=1)
    )

    # -- For malware sampling, comment out phishing_sample above and uncomment:
    # malware_sample = (
    #     malware_df[["domain_name"]]
    #     .sample(n=100_000, random_state=42)
    #     .assign(label=1)
    # )

    # Sample 50,000 benign domains from each source
    benign1_sample = (
        benign1_df[["domain_name"]]
        .sample(n=50_000, random_state=42)
        .assign(label=0)
    )
    benign2_sample = (
        benign2_df[["domain_name"]]
        .sample(n=50_000, random_state=42)
        .assign(label=0)
    )
    benign_sample = pd.concat([benign1_sample, benign2_sample], ignore_index=True)

    # Combine phishing (or malware) and benign samples
    combined_df = pd.concat([phishing_sample, benign_sample], ignore_index=True)
    # -- If using malware instead, replace phishing_sample with malware_sample above

    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    out_path = dataset_dir / "phishing" / "phishing_preprocessed.csv"
    # -- For malware output, change filename:
    # out_path = dataset_dir / "malware" / "malware_preprocessed.csv"
    combined_df.to_csv(out_path, index=False)

    print(f"Created {out_path.name} with shape {combined_df.shape}")

if __name__ == "__main__":
    main()
