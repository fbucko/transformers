# run.py
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, AutoTokenizer
from config import Config
from data.dataset import DatasetLoader, DomainDataset
from models.transformer_model import get_model
from training.trainer import train

def main():
    # 1. Load the dataset
    loader = DatasetLoader(Config.DATA_PATH)
    domains, labels = loader.load()
    print(f"Loaded {len(domains)} records.")

    # 2. Split data into training and validation sets
    train_domains, val_domains, train_labels, val_labels = train_test_split(
        domains, labels, test_size=0.25, random_state=42
    )

    # 3. Loop over each transformer-based model defined in the config
    config = Config()
    for model_name in Config.MODEL_NAMES:
        print(f"\n=== Training model: {model_name} ===\n")
        # Update the config for the current model
        config.PRETRAINED_MODEL_NAME = model_name
        config.MODEL_SAVE_NAME = model_name.replace('/', '-')

        # 4. Initialize the tokenizer using the current model name.
        # tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 5. Create dataset objects for the current tokenizer
        train_dataset = DomainDataset(train_domains, train_labels, tokenizer, Config.MAX_LENGTH)
        val_dataset = DomainDataset(val_domains, val_labels, tokenizer, Config.MAX_LENGTH)
        
        # 6. Start distributed training for the current model
        train(config, train_dataset, val_dataset, tokenizer, get_model)

if __name__ == "__main__":
    main()
