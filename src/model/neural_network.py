import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
import torchmetrics
from tqdm import tqdm


def initialize_final_layer_bias_with_class_weights(model, weights):
    for name, param in model.ff_layer.named_parameters():
        final_layer_bias_name = list(dict(model.ff_layer.named_parameters()).keys())[-1]
        assert "bias" in final_layer_bias_name
        
        if name == final_layer_bias_name:    
            assert len(param) == 3
            param.data[0] = weights.loc[0]
            param.data[1] = weights.loc[1]
            param.data[2] = weights.loc[2]


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