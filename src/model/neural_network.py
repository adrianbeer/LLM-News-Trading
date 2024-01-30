import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
import torchmetrics
from tqdm import tqdm

class BERTClassifier(pl.LightningModule):
    
    def __init__(self, 
                 bert_model_name, 
                 num_classes, 
                 deactivate_bert_learning, 
                 learning_rate,
                 class_weights):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        class_weights = torch.Tensor(class_weights, device=self.device)
        self.register_buffer("class_weights", class_weights)
        
        average = "weighted"
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, average=average)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes, average=average)
        self.train_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=average)
        self.val_f1_score = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=average)
        
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
        
    def forward(self, bert_args):
        outputs = self.bert(input_ids=bert_args[0], attention_mask=bert_args[1])
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.ff_layer(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def cross_entropy_loss(self, logits, labels, weights=None):
        return F.cross_entropy(logits, labels, weight=weights)

    def training_step(self, train_batch, batch_idx):
        bert_args = (train_batch["input_id"], train_batch["mask"])
        y = train_batch["target"]
        logits = self.forward(bert_args)
        preds = logits.softmax(dim=1)
        
        weighted_loss = self.cross_entropy_loss(logits, y, self.class_weights)
        unweighted_loss = self.cross_entropy_loss(logits, y)
        self.train_accuracy(preds, y)
        self.train_f1_score(preds, y)
    
        self.log_dict({'train_loss (weighted)': weighted_loss,
                       'train_loss': unweighted_loss,
                       "train_f1_score": self.train_f1_score,
                       "train_accuracy": self.train_accuracy}, 
                      on_step=True, on_epoch=True, prog_bar=True)
        return weighted_loss
    
    def validation_step(self, val_batch, batch_idx):
        bert_args = (val_batch["input_id"], val_batch["mask"])
        y = val_batch["target"]
        logits = self.forward(bert_args)
        preds = logits.softmax(dim=1)
        
        loss = self.cross_entropy_loss(logits, y)        
        self.val_accuracy(preds, y)
        self.val_f1_score(preds, y)
    
        self.log_dict({'val_loss': loss,
                       "val_f1_score": self.val_f1_score,
                       "val_accuracy": self.val_accuracy})
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

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