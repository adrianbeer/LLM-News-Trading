import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as pl
from torchmetrics.regression import MeanAbsoluteError
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

class BERTRegressor(pl.LightningModule):
    
    def __init__(self, 
                 bert_model_name, 
                 deactivate_bert_learning, 
                 learning_rate,
                 dropout_rate,
                 hidden_layer_size):
        super().__init__()
        self.save_hyperparameters()
        
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3)
        return optimizer, scheduler

    def l1_loss(self, preds, labels):
        return F.l1_loss(preds, labels)

    def training_step(self, train_batch, batch_idx):
        y = train_batch["target"]
        preds = self.forward(train_batch)
        loss = self.train_accuracy(preds, y)

        self.log_dict({"train/loss": loss}, on_step=True, on_epoch=True, prog_bar=True)
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
        preds = self.forward(val_batch)
        self.validation_step_outputs.append(preds)
        
        loss = self.val_accuracy(preds, y)
    
        self.log_dict({
            'val/loss': loss
            })
        
        return preds
    
    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs

        # dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        
        # model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        # torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        # artifact = wandb.Artifact(name="model.ckpt", type="model")
        # artifact.add_file(model_filename)
        # self.logger.experiment.log_artifact(artifact)

        flattened_preds = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/preds": wandb.Histogram(flattened_preds.to("cpu")),
            "global_step": self.global_step})
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
