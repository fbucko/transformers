"""
Author: Filip Bucko
Email: xbucko05@vutbr.cz
Institution: Brno University of Technology - Faculty of Information Technology
Date: May 19, 2025
Description:
    Builds a balanced DGA vs. benign domain dataset by sampling and labeling.
    Automatically locates the project's `datasets` folder regardless of where the script is run.
"""

import pandas as pd
from pathlib import Path

def find_project_root() -> Path:
    """
    Walk up from this script's directory until a 'datasets' folder is found.
    Returns the Path to the project root containing 'datasets'.
    """
    current = Path(__file__).resolve().parent
    while True:
        if (current / "datasets").exists():
            return current
        if current.parent == current:
            raise RuntimeError("Could not find 'datasets' directory in any parent path.")
        current = current.parent

def get_benign_sample(datasets_dir: Path) -> pd.DataFrame:
    """
    Load two benign CSVs, sample 50k domains from each, label them 0, and concatenate.
    """
    # Read source files
    benign1 = pd.read_csv(datasets_dir / "benign" /"benign_2312_anonymized.csv")
    benign2 = pd.read_csv(datasets_dir / "benign" / "umbrella_benign_FINISHED.csv")

    # Sample and label
    b1 = benign1[["domain_name"]].sample(n=50_000, random_state=42).copy()
    b1["label"] = 0
    b2 = benign2[["domain_name"]].sample(n=50_000, random_state=42).copy()
    b2["label"] = 0

    # Combine both benign samples
    return pd.concat([b1, b2], ignore_index=True)

def get_dga_sample(datasets_dir: Path) -> pd.DataFrame:
    """
    Load DGA CSV, sample 100k domains, and label them 1.
    """
    dga = pd.read_csv(datasets_dir / "dga" / "dga_2310.csv")
    sample = dga[["domain_name"]].sample(n=100_000, random_state=42).copy()
    sample["label"] = 1
    return sample

def create_dga_dataset(datasets_dir: Path) -> pd.DataFrame:
    """
    Combine the DGA sample and benign sample, shuffle, and return.
    """
    dga_df    = get_dga_sample(datasets_dir)
    benign_df = get_benign_sample(datasets_dir)
    combined  = pd.concat([dga_df, benign_df], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    # Locate the project root and datasets directory
    project_root = find_project_root()
    datasets_dir = project_root / "datasets"

    # Build and save the DGA vs. benign dataset
    dga_df = create_dga_dataset(datasets_dir)
    out_path = datasets_dir / "dga" /"dga_preprocessed.csv"
    dga_df.to_csv(out_path, index=False)

    print(f"Created {out_path.name} with shape {dga_df.shape}")

if __name__ == "__main__":
    main()
