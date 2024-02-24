import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
import torchmetrics
from tqdm import tqdm
from src.config import MODEL_CONFIG
from src.model.bert_classifier import (
    initialize_final_layer_bias_with_class_weights,
)

def get_model(ckpt, model_args, dm) -> pl.LightningModule:
    if ckpt:
        #! I found that when you integrate lightning with ray tune, you must use model.load_state_dict() instead of model.load_from_checkpoint() to really get the trained weights.
        model: pl.LightningModule = MODEL_CONFIG.neural_net.load_from_checkpoint(ckpt, 
                                                                                 **model_args)
        print(f"Using Checkpointed model at {ckpt}...")
    elif MODEL_CONFIG.task == "Regression":
        print("Initialize news regression model...")
        model: pl.LightningModule = MODEL_CONFIG.neural_net(bert_model_name=MODEL_CONFIG.pretrained_network,
                                                  **model_args)
    elif MODEL_CONFIG.task == "Classification":
        print("Initialize new Classification model...")
        dm.setup("fit")
        class_distribution = dm.get_class_distribution()
        print(dm.train_dataloader().dataset.get_class_distribution())
        model: pl.LightningModule = MODEL_CONFIG.neural_net(bert_model_name=MODEL_CONFIG.pretrained_network,
                                        num_classes=3,
                                        class_weights=1 / class_distribution.values,
                                        **model_args)
        initialize_final_layer_bias_with_class_weights(model, class_distribution)
    else:
        raise ValueError()
    return model


@torch.no_grad
def predict_cls(model, dataloader, device):
    model.eval()
    outputs = None
    for batch in tqdm(dataloader, desc="predict_cls"):
        batch_inputs, batch_masks = tuple(b.to(device) for b in batch)
        output = model(batch_inputs, attention_mask=batch_masks)
        cls_tokens = np.array(output.last_hidden_state[:,0,:].tolist())
        if outputs is None:
            outputs = cls_tokens
        else:
            outputs = np.concatenate([outputs, cls_tokens])
    return outputs


@torch.no_grad
def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        output += model(batch_inputs, batch_masks).view(1,-1).tolist()[0]
    return np.array(output)