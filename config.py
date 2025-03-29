# config.py
class Config:
    # DATA_PATH = "../datasets/dga_preprocessed.csv"
    # DATA_PATH = "../datasets/phishing_preprocessed.csv"
    DATA_PATH = "../datasets/malware_preprocessed.csv"
    MAX_LENGTH = 32
    # BATCH_SIZE = 512
    # BATCH_SIZE = 256    
    BATCH_SIZE = 128    
    EPOCHS = 20
    LEARNING_RATE = 5e-5
    DISTRIBUTED_PORT = 23456
    # TYPE = "dga"
    # TYPE = "phish"
    TYPE = "malware"
    # Default model (will be overridden in the loop)
    PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
    MODEL_SAVE_NAME = ""
    # List of transformer-based models to try
    MODEL_NAMES = [
        # "distilbert-base-uncased",
        # "distilbert-base-cased",
        # "bert-base-uncased",
        # "bert-base-cased",
        # "prajjwal1/bert-tiny",
        # "prajjwal1/bert-mini",
        # "prajjwal1/bert-small",
        # "prajjwal1/bert-medium",
        "albert-base-v2",
        # "google/electra-small-discriminator",
        # "google/electra-base-discriminator",
        # "google/mobilebert-uncased"
    ]
    
        # "distilbert-base-uncased", - done 512
        # "distilbert-base-cased", - done 512
        # "bert-base-uncased", - not trained
        # "bert-base-cased", - not trained
        # "prajjwal1/bert-tiny", - done 512
        # "prajjwal1/bert-mini", - done 512
        # "prajjwal1/bert-small", - done 512
        # "prajjwal1/bert-medium", - done 512
        # "albert-base-v2", - done 128
        # "google/electra-small-discriminator",- done 512
        # "google/electra-base-discriminator", - done 256
        # "google/mobilebert-uncased" - done - more epochs 256
    
