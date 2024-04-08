import torch
import torch.nn as nn
from transformers import AutoModel
import lightning as pl
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torch.optim.lr_scheduler import LambdaLR, LinearLR, MultiStepLR
from lightning.pytorch.utilities import grad_norm

class NNRegressor(pl.LightningModule):
    
    def __init__(self, 
                 base_model, 
                 deactivate_bert_learning, 
                 learning_rate,
                 dropout_rate,
                 hidden_layer_size,
                 n_warm_up_epochs):
        super().__init__()
        self.save_hyperparameters()
        
        # Plotting and Visualizaton
        self.validation_outputs = []
        self.validation_labels = []
        self.training_outputs = []
        self.training_labels = []
                
        self.train_mae = MeanAbsoluteError()
        self.val_loss = MeanAbsoluteError()

        self.bert: nn.Module = AutoModel.from_pretrained(base_model)
        
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

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.ff_layer, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                     lr=self.hparams.learning_rate, 
                                     weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00005, steps_per_epoch=52000, epochs=3)
        
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

    def loss_function(self, preds, y, weights=None):
        loss = self.weighted_mse_loss(preds=preds, y=y, weights=weights)
        return loss

    def weighted_mse_loss(self, preds, y, weights=None):
        # cant raise to a fraction, otherwise NaNs occurr for negative values (sqrt of neg. value)
        loss = (preds - y) ** 2 
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        y = train_batch["target"]
        preds = self.forward(train_batch)
        weights = train_batch["sample_weights"]
        loss = self.loss_function(preds, y, weights)
        
        self.train_mae(preds, y)

        self.training_outputs.append(preds)
        self.training_labels.append(y)

        # import pdb
        # pdb.set_trace()

        self.log_dict({
            "train/loss": loss,
            "train/mae": self.train_mae
                       }, 
                      on_step=True, 
                      on_epoch=True, 
                      prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        y = val_batch["target"]
        preds = self.forward(val_batch)
        # import pdb
        # pdb.set_trace()
        self.validation_outputs.append(preds)
        self.validation_labels.append(y)
        
        loss = self.loss_function(preds, y)
    
        self.log_dict({
            f"val/loss": loss
            })
        
        return preds
    
    def calculate_quantile_filteted_acc(self, flattened_preds, flattened_labels, percent):
        # .float() required as the quantile function can't handle bfloat16 dtype
        q = torch.quantile(flattened_preds.float(), percent)
        mask = torch.where(flattened_preds >= q, 1, 0)
        quantile_filtered_acc = torch.mean((torch.sign(flattened_preds[torch.nonzero(mask)]) == torch.sign(flattened_labels[torch.nonzero(mask)])).float())
        return quantile_filtered_acc
    
    def on_train_epoch_end(self):    
        flattened_labels = torch.flatten(torch.cat(self.training_labels))
        flattened_preds = torch.flatten(torch.cat(self.training_outputs))
        
        self.log_dict(
            {
                "train/quantile_95_filtered_acc": self.calculate_quantile_filteted_acc(flattened_preds, flattened_labels, 0.95),
                "train/quantile_50_filtered_acc": self.calculate_quantile_filteted_acc(flattened_preds, flattened_labels, 0.5),
                "global_step": self.global_step
            })
        
        # import pdb
        # pdb.set_trace()
        
        flattened_preds = flattened_preds.to("cpu")
        if not torch.isnan(flattened_preds).any():
            self.logger.experiment.add_histogram("train/preds", flattened_preds, self.current_epoch)
        else:
            print("NaN flattened_preds in train/preds")
        self.training_outputs.clear()
        self.training_labels.clear()
    
    def on_validation_epoch_end(self):    
        flattened_labels = torch.flatten(torch.cat(self.validation_labels))
        flattened_preds = torch.flatten(torch.cat(self.validation_outputs))
        self.log_dict(
            {
                "val/quantile_95_filtered_acc": self.calculate_quantile_filteted_acc(flattened_preds, flattened_labels, 0.95),
                "val/quantile_50_filtered_acc": self.calculate_quantile_filteted_acc(flattened_preds, flattened_labels, 0.5),
                "global_step": self.global_step
            })        

        # import pdb
        # pdb.set_trace()
        flattened_preds = flattened_preds.to("cpu")
        if not torch.isnan(flattened_preds).any():
            self.logger.experiment.add_histogram("val/preds", flattened_preds, self.current_epoch)
        else:
            print("NaN preds in val/preds")
 
        self.validation_outputs.clear()
        self.validation_labels.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


# def lmbda(epoch, max_epochs, warmup):
#     lr_factor = 1 - 0.5 * epoch / max_epochs
#     if epoch <= warmup:
#         lr_factor = epoch * 1.0 / warmup
#     return lr_factor

# class LinearWarmupScheduler(optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, warmup, max_epochs):
#         self.warmup = warmup
#         self.max_epochs = max_epochs
#         super().__init__(optimizer)
#         print(f"{max_epochs=}")

#     def get_lr(self):
#         lr_factor = self.get_lr_factor(epoch=self.last_epoch)
#         return [group['lr'] * lr_factor for group in self.optimizer.param_groups]

#     def get_lr_factor(self, epoch):
#         lr_factor = 1 - 0.5 * epoch / self.max_epochs
#         if epoch <= self.warmup:
#             lr_factor = epoch * 1.0 / self.warmup
#         return lr_factor