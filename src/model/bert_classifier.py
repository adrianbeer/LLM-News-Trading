import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
import torchmetrics
from transformers import AutoModel


class BERTClassifier(pl.LightningModule):
    
    def __init__(self, 
                 base_model, 
                 num_classes, 
                 deactivate_bert_learning, 
                 learning_rate,
                 dropout_rate,
                 hidden_layer_size,
                 n_warm_up_epochs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # class_weights = torch.Tensor(class_weights, device=self.device)
        # self.register_buffer("class_weights", class_weights)
        
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.bert: nn.Module = AutoModel.from_pretrained(base_model)
        
        if deactivate_bert_learning:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        dropout_rate = self.hparams.dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        hls = self.hparams.hidden_layer_size
        self.ff_layer: nn.Module = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hls),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hls, hls),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hls, num_classes)
        )
        
    def forward(self, input_ids, masks):
        outputs = self.bert(input_ids=input_ids, attention_mask=masks)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.ff_layer(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                     lr=self.hparams.learning_rate, 
                                     weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=52000, epochs=4)
        
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            "frequency": 1,
            "monitor": f"train/loss"
        }
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler_config 
            }


    def cross_entropy_loss(self, logits, labels, weights=None):
        loss = F.cross_entropy(logits, labels, reduction='none')
        if weights is not None:
            loss = loss * weights
        loss = torch.mean(loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        y = train_batch["target"]
        logits = self.forward(train_batch["input_id"], train_batch["mask"])
        preds = logits.softmax(dim=1)
        
        weights = train_batch['sample_weights']
        unweighted_loss = self.cross_entropy_loss(logits, y, weights=weights)
        self.train_accuracy(preds, y)
        # self.train_f1_score(preds, y)
    
        self.log_dict({'train/loss': unweighted_loss,
                    #    "train/f1_score": self.train_f1_score,
                       "train/acc": self.train_accuracy}, 
                      on_step=True, on_epoch=True, prog_bar=False)
        
        return unweighted_loss
    
    def validation_step(self, val_batch, batch_idx):
        y = val_batch["target"]
        logits = self.forward(val_batch["input_id"], val_batch["mask"])
        preds = logits.softmax(dim=1)
        
        loss = self.cross_entropy_loss(logits, y)        
        self.val_accuracy(preds, y)
        # self.val_f1_score(preds, y)
    
        self.log_dict({'val/loss': loss,
                    #    "val/f1_score": self.val_f1_score,
                       "val/acc": self.val_accuracy})
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["input_id"], batch["mask"])


# def initialize_final_layer_bias_with_class_weights(model, weights):
#     for name, param in model.ff_layer.named_parameters():
#         final_layer_bias_name = list(dict(model.ff_layer.named_parameters()).keys())[-1]
#         assert "bias" in final_layer_bias_name
        
#         if name == final_layer_bias_name:    
#             assert len(param) == 3
#             param.data[0] = weights.loc[0]
#             param.data[1] = weights.loc[1]
            # param.data[2] = weights.loc[2]