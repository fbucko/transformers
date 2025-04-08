class Config:
    # Data parameters
    # For domain names, set DATA_TYPE = "domain"
    # For RDAP data, set DATA_TYPE = "rdap"
    DATA_TYPE = "domain"  # or "rdap"
    
    # Paths for each data type
    # Adjust these paths as needed
    TYPE = "dga"  # or "malware", "phishing"
    DOMAIN_DATA_PATH = "../datasets/dga_preprocessed.csv"
    RDAP_DATA_PATH = "../datasets/phishing/rdap_phishing_preprocessed.csv"
#     RDAP_DATA_PATH = "../datasets/malware/rdap_malware_preprocessed.csv"
    
    # Other parameters remain the same
    MAX_LENGTH = 64
    BATCH_SIZE = 512    
    EPOCHS = 20
    LEARNING_RATE = 5e-5
    DISTRIBUTED_PORT = 23456

    PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
    MODEL_SAVE_NAME = ""
    MODEL_NAMES = [
         "distilbert-base-uncased", 
    ]
