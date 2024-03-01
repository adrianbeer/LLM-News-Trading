import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
from torchmetrics.regression import MeanAbsoluteError
from torch.optim.lr_scheduler import LambdaLR
import wandb
from torch import optim
import numpy as np
from lightning.pytorch.utilities import grad_norm

class BERTRegressor(pl.LightningModule):
    
    def __init__(self, 
                 bert_model_name, 
                 deactivate_bert_learning, 
                 learning_rate,
                 dropout_rate,
                 hidden_layer_size):
        super().__init__()
        self.save_hyperparameters()
        
        # Plotting and Visualizaton
        self.validation_step_outputs = []
        
        self.train_accuracy = MeanAbsoluteError()
        self.val_accuracy = MeanAbsoluteError()

        self.bert: nn.Module = BertModel.from_pretrained(bert_model_name)
        
        if self.hparams.deactivate_bert_learning:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        hls = self.hparams.hidden_layer_size
        
        self.ff_layer: nn.Module = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hls), nn.LeakyReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            
            nn.Linear(hls, hls), nn.LeakyReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            
            nn.Linear(hls, hls), nn.LeakyReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            
            nn.Linear(hls, 1) # Output Layer
        )
        
    def forward(self, batch):
        outputs = self.bert(input_ids=batch["input_id"], attention_mask=batch["mask"])
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        preds: nn.Tensor = self.ff_layer(x)
        preds.squeeze_(1)
        return preds

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.ff_layer, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.learning_rate, 
                                     weight_decay=0.01)
        scheduler = LinearWarmupScheduler(optimizer=optimizer, 
                                          warmup=4, 
                                          max_epochs=self.trainer.max_epochs)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler,
            "monitor": "train/loss"
            }

    def l1_loss(self, preds, labels):
        return F.l1_loss(preds, labels)

    def training_step(self, train_batch, batch_idx):
        y = train_batch["target"]
        preds = self.forward(train_batch)
        loss = self.train_accuracy(preds, y)

        self.log_dict({"train/loss": loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        y = val_batch["target"]
        preds = self.forward(val_batch)
        self.validation_step_outputs.append(preds)
        
        loss = self.val_accuracy(preds, y)
    
        self.log_dict({
            'val/loss': loss
            })
        
        return preds
    
    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        try:
            flattened_preds = torch.flatten(torch.cat(validation_step_outputs)).to("cpu")
            self.logger.experiment.log(
                {"valid/preds": wandb.Histogram(flattened_preds),
                "global_step": self.global_step})
            
            # data = [[s] for s in flattened_preds.numpy()]
            # table = wandb.Table(data=data, columns=["predictions"])
            # self.logger.experiment.log({
            #     'my_histogram': wandb.plot.histogram(table, 
            #                                         "predictions",
            #                                         title="Predictions")
            #     })
        except Exception as e:
            # Logging this failes sometimes for unknown reasons
            print(e)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


class LinearWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_epochs):
        self.warmup = warmup
        self.max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 1 - 0.5 * epoch / self.max_epochs
        if epoch <= self.warmup:
            lr_factor = epoch * 1.0 / self.warmup
        return lr_factor