# run.py
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
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

    # 3. Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(Config.PRETRAINED_MODEL_NAME)

    # 4. Create dataset objects
    train_dataset = DomainDataset(train_domains, train_labels, tokenizer, Config.MAX_LENGTH)
    val_dataset = DomainDataset(val_domains, val_labels, tokenizer, Config.MAX_LENGTH)

    # 5. Start training using distributed processes
    train(Config(), train_dataset, val_dataset, tokenizer, get_model)

if __name__ == "__main__":
    main()
