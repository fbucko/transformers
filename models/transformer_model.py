# transformers/models/transformer_model.py
from transformers import AutoModelForSequenceClassification
from adapters      import AutoAdapterModel, AdapterConfig


def get_model_standard(model_name: str, num_labels: int = 2):
    """
    Standard full fine‑tuning: loads a sequence‑classification model
    with all weights trainable.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )


def get_model_adapters(model_name: str, num_labels: int = 2):
    """
    Adapter‑based fine‑tuning: load the base model, add a Pfeiffer adapter,
    add a classification head, then freeze everything except the adapter.
    """
    # 1) Load the base transformer (no head attached)
    model = AutoAdapterModel.from_pretrained(model_name)

    # 2) Attach a Pfeiffer-style adapter
    adapter_cfg = AdapterConfig.load(
        "pfeiffer",           # Pfeiffer recipe
        reduction_factor=16,   # bottleneck = hidden_size/16
        non_linearity="relu"
    )
    model.add_adapter("malicious_domain", config=adapter_cfg)

    # 3) Add a task-specific classification head (positional head_name)
    model.add_classification_head(
        "malicious_domain",
        num_labels=num_labels
    )

    # 4) Activate and train only the adapter + head
    model.set_active_adapters("malicious_domain")
    model.train_adapter("malicious_domain")

    return model