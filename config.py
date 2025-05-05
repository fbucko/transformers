# config.py

class Config:
    # Path to your CSV dataset
    # DATA_PATH = "../datasets/dga/dga_preprocessed.csv"
    DATA_PATH = "../datasets/phishing/phishing_preprocessed.csv"
    # DATA_PATH = "../datasets/malware/malware_preprocessed.csv"
    
    # Tokenizer / model settings
    MAX_LENGTH = 32

    # Training hyper-parameters
    EPOCHS = 40
    # BATCH_SIZE = 512
    # BATCH_SIZE = 256    
    # BATCH_SIZE = 128
    
    # ----------- distilbert-base-uncased & similar bigger models -----------
    # BATCH_SIZE = 512
    # LEARNING_RATE  = 3.946212980759097e-05 # DGA, Phishing, Malware 
    # WEIGHT_DECAY   = 0.09266588657937942 # DGA, Phishing, Malware
    # WARMUP_RATIO   = 0.1454543991712842 # DGA, Phishing, Malware 
    # LR_SCHEDULER   = "linear"   
    
    # ----------- bert-small & similar medium models -----------
    # BATCH_SIZE = 512
    # LEARNING_RATE  = 3.946212980759097e-05 # DGA, Phishing, Malware 
    # WEIGHT_DECAY   = 0.09266588657937942 # DGA, Phishing, Malware
    # WARMUP_RATIO   = 0.1454543991712842 # DGA, Phishing, Malware 
    # LR_SCHEDULER   = "linear"    
    
    # ----------- bert-tiny & similar small models  -----------
    BATCH_SIZE = 128
    LEARNING_RATE  = 4.749974771378411e-05 # DGA, Phishing, Malware 
    WEIGHT_DECAY   = 0.019871568153417243 # DGA, Phishing, Malware
    WARMUP_RATIO   = 0.00110442342472048 # DGA, Phishing, Malware 
    LR_SCHEDULER   = "linear"    
    

    # Distributed training
    DISTRIBUTED_PORT = 23456

    # Task-specific identifiers
    # TYPE = "dga"
    TYPE = "phish"
    # TYPE             = "malware"                      # used in log/checkpoint names
    PRETRAINED_MODEL_NAME = "distilbert-base-uncased" # Default model (will be overridden in the loop)
    MODEL_SAVE_NAME       = ""                        # will be set per-model in run.py

    # Which models to loop over in run.py
    MODEL_NAMES = [
            # "distilbert-base-uncased",
            # "distilbert-base-cased",
            "prajjwal1/bert-tiny",
            "prajjwal1/bert-mini",
            # "prajjwal1/bert-small",
            # "prajjwal1/bert-medium",
            "albert-base-v2",
            # "google/electra-small-discriminator",
            "google/electra-base-discriminator",
            "google/mobilebert-uncased"
        ]

    
    # Early-stopping: how many epochs with no improvement before stopping
    PATIENCE = 5

    # Which metric to monitor for “best checkpoint”: "loss" or "f1"
    CKPT_MONITOR = "loss"


    
# "distilbert-base-uncased", - done 512
# "distilbert-base-cased", - done 512
# "prajjwal1/bert-tiny", - done 512
# "prajjwal1/bert-mini", - done 512
# "prajjwal1/bert-small", - done 512
# "prajjwal1/bert-medium", - done 512
# "albert-base-v2", - done 128
# "google/electra-small-discriminator",- done 512
# "google/electra-base-discriminator", - done 256
# "google/mobilebert-uncased" - done - more epochs 256

# DGA
# "distilbert-base-uncased", - done 512
# "distilbert-base-cased", - done 512
# "prajjwal1/bert-tiny", - done 128
# "prajjwal1/bert-mini", - done 128
# "prajjwal1/bert-small", - done 512
# "prajjwal1/bert-medium", - done 512
# "albert-base-v2", - done 128
# "google/electra-small-discriminator",- done 512
# "google/electra-base-discriminator", - done 256
# "google/mobilebert-uncased" - done 256

# Malware
# "distilbert-base-uncased", - done 512
# "distilbert-base-cased", - done 512
# "prajjwal1/bert-tiny", -  done 128
# "prajjwal1/bert-mini", -  done 128
# "prajjwal1/bert-small", - done 512
# "prajjwal1/bert-medium", - done 512
# "albert-base-v2", -  128
# "google/electra-small-discriminator",- done 512
# "google/electra-base-discriminator", - done 128
# "google/mobilebert-uncased" - done 128

# Phishing
# "distilbert-base-uncased", - done 512
# "distilbert-base-cased", - done 512
# "prajjwal1/bert-tiny", -128
# "prajjwal1/bert-mini", -128
# "prajjwal1/bert-small", - done 512
# "prajjwal1/bert-medium", - done 512
# "albert-base-v2", - 128
# "google/electra-small-discriminator",- done 512
# "google/electra-base-discriminator", - 128
# "google/mobilebert-uncased" - 128