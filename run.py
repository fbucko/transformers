import argparse
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from config import Config
from data.dataset import DatasetLoader, DomainDataset, RDAPDataset, DNSDataset
from models.transformer_model import get_model_standard, get_model_adapters
from training.trainer import train

def parse_args(config: Config):
    parser = argparse.ArgumentParser(
        description="Train a transformer model on domain or RDAP data."
    )
    parser.add_argument("--data_type", type=str, choices=["domain", "rdap", "dns"],
                        default=config.DATA_TYPE,
                        help="Type of data to train on: 'domain' for domain names or 'rdap' for RDAP data.")
    parser.add_argument("--type", type=str, choices=["dga", "malware", "phish"],
                        default=config.TYPE,
                        help="Type of data to train on: 'domain' for domain names or 'rdap' for RDAP data.")
    parser.add_argument("--domain_data_path", type=str,
                        default=config.DOMAIN_DATA_PATH,
                        help="Path to the domain dataset CSV.")
    parser.add_argument("--rdap_data_path", type=str,
                        default=config.RDAP_DATA_PATH,
                        help="Path to the RDAP dataset CSV.")
    parser.add_argument("--dns_data_path", type=str,
                        default=config.DNS_DATA_PATH,
                        help="Path to the RDAP dataset CSV.")
    parser.add_argument("--pretrained_model_name", type=str,
                        default=config.PRETRAINED_MODEL_NAME,
                        help="Pretrained transformer model name (e.g., 'distilbert-base-uncased').")
    parser.add_argument("--max_length", type=int,
                        default=config.MAX_LENGTH,
                        help="Maximum token length for the model.")
    parser.add_argument("--batch_size", type=int,
                        default=config.BATCH_SIZE,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int,
                        default=config.EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float,
                        default=config.LEARNING_RATE,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--distributed_port", type=int,
                        default=config.DISTRIBUTED_PORT,
                        help="Port for distributed training initialization.")
    parser.add_argument(
        "--use_adapters",
        action="store_true",
        help="If set, use adapter-based fine-tuning instead of full fine-tuning."
    )
    return parser.parse_args()

def main():
    # Instantiate your configuration object.
    config = Config()

    # Parse command-line arguments, using the config values as defaults.
    args = parse_args(config)
    
    # Update the config object with any values provided from the command line.
    config.DATA_TYPE = args.data_type
    if args.data_type == "domain":
        config.DATA_PATH = args.domain_data_path
    elif args.data_type == "rdap":
        config.DATA_PATH = args.rdap_data_path
    elif args.data_type == "dns":
        config.DATA_PATH = args.dns_data_path
    else:
        pass
    
    config.TYPE = args.type
    config.PRETRAINED_MODEL_NAME = args.pretrained_model_name
    config.MAX_LENGTH = args.max_length
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.DISTRIBUTED_PORT = args.distributed_port

    # Load the dataset using the updated config.
    loader = DatasetLoader(config.DATA_PATH, data_type=config.DATA_TYPE)
    df = loader.load()
    print(f"Loaded {len(df)} records.")
    print(df.head())

    # Split the dataset into training and validation sets.
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

    # Initialize the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    
    # Create Dataset objects based on the data type.
    if config.DATA_TYPE == "domain":
        train_dataset = DomainDataset(train_df["domain_name"], train_df["label"], tokenizer, config.MAX_LENGTH)
        val_dataset = DomainDataset(val_df["domain_name"], val_df["label"], tokenizer, config.MAX_LENGTH)
    elif config.DATA_TYPE == "rdap": 
        train_dataset = RDAPDataset(train_df, tokenizer, config.MAX_LENGTH)
        val_dataset = RDAPDataset(val_df, tokenizer, config.MAX_LENGTH)
    elif config.DATA_TYPE == "dns": 
        train_dataset = DNSDataset(train_df, tokenizer, config.MAX_LENGTH)
        val_dataset = DNSDataset(val_df, tokenizer, config.MAX_LENGTH)
    
     # Pick your model‑factory
    if args.use_adapters:
        get_model_fn = get_model_adapters
        config.USE_ADAPTERS = True
    else:
        get_model_fn = get_model_standard

    # Inspect a few examples
    # num_samples_to_check = 10  # You can adjust this number

    # for i in range(num_samples_to_check):
    #     sample = train_dataset[i]
    #     # Convert token IDs back to tokens for a human-readable view.
    #     tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].tolist())
    #     print(f"Sample {i+1}:")
    #     print("Tokens:", tokens)
    #     print("Tokens:", len(tokens))
    #     print("Attention Mask:", sample['attention_mask'].tolist())
    #     print("Label:", sample['label'].item())
    #     print("-" * 50)
    
    # Start the training process, passing the config object along.
    train(config, train_dataset, val_dataset, tokenizer, get_model_fn)

if __name__ == "__main__":
    main()
