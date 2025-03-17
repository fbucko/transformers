# config.py

class Config:
    TYPE = "dga"
    
    # Data parameters
    # DATA_PATH = "../datasets/phishing_preprocessed.csv"
    DATA_PATH = "../datasets/dga_preprocessed.csv"
    
    # Model and tokenizer parameters
    PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 32
    
    # Training parameters
    BATCH_SIZE = 512
    EPOCHS = 25
    LEARNING_RATE = 5e-5

    # Distributed training settings
    DISTRIBUTED_PORT = "23456"
