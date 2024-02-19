import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
import torchmetrics
from tqdm import tqdm





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