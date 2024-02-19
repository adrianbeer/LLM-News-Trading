import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
from torchmetrics.regression import MeanAbsoluteError


class BERTRegressor(pl.LightningModule):
    
    def __init__(self, 
                 bert_model_name, 
                 deactivate_bert_learning, 
                 learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.train_accuracy = MeanAbsoluteError()
        self.val_accuracy = MeanAbsoluteError()

        self.bert: nn.Module = BertModel.from_pretrained(bert_model_name)
        if deactivate_bert_learning:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(0)
        
        self.ff_layer: nn.Module = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size, 10),
            nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(10, 1) # Output Layer
        )
        
    def forward(self, input_ids, masks):
        outputs = self.bert(input_ids=input_ids, attention_mask=masks)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        preds = self.ff_layer(x)
        preds.squeeze_(1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_loss(self, preds, labels):
        return F.l1_loss(preds, labels)

    def training_step(self, train_batch, batch_idx):
        y = train_batch["target"]

        preds = self.forward(train_batch["input_id"], train_batch["mask"])

        loss = self.train_accuracy(preds, y)

        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    def custom_histogram_adder(self):    
        for name, params in self.ff_layer.named_parameters():
            self.logger.experiment.add_histogram(name,
                                                 params,
                                                 self.current_epoch)    

    # def training_epoch_end(self,outputs): 
    #     # calculating average loss  
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
 
    #     # logging histograms
    #     self.custom_histogram_adder()
         
    #     epoch_dictionary={            
    #                       'loss': avg_loss
    #                       }
 
    #     return epoch_dictionary
    
    def validation_step(self, val_batch, batch_idx):
        y = val_batch["target"]
        preds = self.forward(val_batch["input_id"], val_batch["mask"])
        
        loss = self.val_accuracy(preds, y)
    
        self.log_dict({
            'val_loss': loss
            })
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["input_id"], batch["mask"])