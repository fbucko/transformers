import pandas as pd

def main():
    
    # Load datasets
    dataset_dir = "../datasets"
    phishing_df = pd.read_csv(f"{dataset_dir}/phishing_strict_2024.csv")  # contains 'domain_name' and 'URL'
    benign1_df = pd.read_csv(f"{dataset_dir}/benign_2312_anonymized.csv")   # contains 'domain_name'
    benign2_df = pd.read_csv(f"{dataset_dir}/umbrella_benign_FINISHED.csv")  # contains 'domain_name'

    # Randomly sample 100,000 phishing domains (using only the 'domain_name' column)
    phishing_sample = phishing_df[['domain_name']].sample(n=100000, random_state=42).copy()
    phishing_sample['label'] = 1

    # Randomly sample 50,000 benign domains from each benign dataset
    benign1_sample = benign1_df[['domain_name']].sample(n=50000, random_state=42).copy()
    benign1_sample['label'] = 0

    benign2_sample = benign2_df[['domain_name']].sample(n=50000, random_state=42).copy()
    benign2_sample['label'] = 0

    # Combine the benign samples (total 100,000 benign domains)
    benign_sample = pd.concat([benign1_sample, benign2_sample], ignore_index=True)

    # Combine phishing and benign samples into one DataFrame
    combined_df = pd.concat([phishing_sample, benign_sample], ignore_index=True)

    # Optionally shuffle the combined dataset (if you want a random mix)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the combined dataset to a CSV file
    combined_df.to_csv(f"{dataset_dir}/phishing_preprocessed.csv", index=False)

    print("Combined dataset created with shape:", combined_df.shape)

if __name__ == "__main__":
    main()
