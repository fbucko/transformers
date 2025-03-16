# config.py

class Config:
    # Data parameters
    DATA_PATH = "../datasets/malware_preprocessed.csv"
    
    # Model and tokenizer parameters
    PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 32
    
    # Training parameters
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 5e-5

    # Distributed training settings
    DISTRIBUTED_PORT = "23456"
