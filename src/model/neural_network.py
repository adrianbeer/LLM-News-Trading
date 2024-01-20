import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import BertModel
import pytorch_lightning as pl
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score


class BERTClassifier(pl.LightningModule):
    
    def __init__(self, bert_model_name, num_classes, deactivate_bert_learning):
        super().__init__()
        self.bert: nn.Module = BertModel.from_pretrained(bert_model_name)
        
        if deactivate_bert_learning:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(0.2)
        self.ff_layer: nn.Module = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, 20),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(20, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.ff_layer(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits,labels)

    def training_step(self, train_batch, batch_idx):
        x, x2, y = train_batch
        logits = self.forward(x, x2)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, x2, y = val_batch
        logits = self.forward(x, x2)
        loss = self.cross_entropy_loss(logits, y)
        
        _, pred_y = torch.max(logits, dim=1)
        for metric_f in [accuracy_score, balanced_accuracy_score]:
            self.log(metric_f.__name__, metric_f(y, pred_y))
        
        self.log('val_loss', loss)
        return loss
    

@torch.no_grad
def evaluate(model: nn.Module, 
             loss_function, 
             validation_dataloader: DataLoader, 
             device,
             tracking_metrics: list,
             is_classification: bool = True):
    model.eval()
    test_loss = []
    
    metrics_batched = dict([(metric.__name__, []) for metric in tracking_metrics])
    
    for batch in validation_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        outputs: Tensor = model(batch_inputs, batch_masks)

        # May have to unsqueeze(1) for regression tasks?
        loss: Tensor = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        
        if is_classification: 
            _, outputs = torch.max(outputs, dim=1)
        
        outputs: np.ndarray = outputs.to('cpu').numpy()
        labels: np.ndarray = batch_labels.to('cpu').numpy()
        
        for metric in tracking_metrics:
            metrics_batched[metric.__name__].append(metric(labels, outputs))

    test_loss = np.mean(test_loss)
    metrics = dict([(name, np.mean(metrics_batched[name])) for name in metrics_batched])
    return test_loss, metrics


@torch.no_grad
def predict_cls(model, dataloader, device):
    model.eval()
    outputs = None
    for batch in dataloader:
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