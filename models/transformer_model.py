# models/transformer_model.py
from transformers import DistilBertForSequenceClassification

def get_model(pretrained_model_name: str, num_labels: int = 2):
    """
    Loads the pre-trained DistilBERT model for sequence classification.
    """
    model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
    return model
