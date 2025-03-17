import pandas as pd


def get_benign_sample(dataset_dir: str) -> pd.DataFrame:
    benign1_df = pd.read_csv(f"{dataset_dir}/benign_2312_anonymized.csv")   # contains 'domain_name'
    benign2_df = pd.read_csv(f"{dataset_dir}/umbrella_benign_FINISHED.csv")  # contains 'domain_name'
    
    # Randomly sample 50,000 benign domains from each benign dataset
    benign1_sample = benign1_df[['domain_name']].sample(n=50000, random_state=42).copy()
    benign1_sample['label'] = 0

    benign2_sample = benign2_df[['domain_name']].sample(n=50000, random_state=42).copy()
    benign2_sample['label'] = 0
    
    # Combine the benign samples (total 100,000 benign domains)
    benign_sample = pd.concat([benign1_sample, benign2_sample], ignore_index=True)
    return benign_sample

def get_dga_sample(dataset_dir: str) -> pd.DataFrame:
    dga_df = pd.read_csv(f"{dataset_dir}/dga_2310.csv")  # contains 'domain_name' and 'URL'
    dga_sample = dga_df[['domain_name']].sample(n=100000, random_state=42).copy()
    dga_sample['label'] = 1
    return dga_sample
    
def create_dga_dataset(datasets_dir: str) -> pd.DataFrame:
    # Combine dga and benign samples into one DataFrame
    combined_df = pd.concat([get_dga_sample(datasets_dir), get_benign_sample(datasets_dir)], ignore_index=True)

    # Shuffle the combined dataset (if you want a random mix)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_df

def main():
    # Load datasets
    datasets_dir = "../datasets"    
    dga_dataset_name = "dga_preprocessed"
    dga_df = create_dga_dataset(datasets_dir)
    # Save the combined dataset to a CSV file
    dga_df.to_csv(f"{datasets_dir}/{dga_dataset_name}.csv", index=False)

    print("Combined dataset created with shape:", dga_df.shape)

if __name__ == "__main__":
    main()
