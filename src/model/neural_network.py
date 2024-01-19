import time

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertModel
from src.utils.time import format_time
import pandas as pd

class MyBertModule(nn.Module):
    
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert: nn.Module = BertModel.from_pretrained(bert_model_name)
        self.ff_layer: nn.Module = None
        
    def deactivate_learning_for_layer(layer: nn.Module):
        for param in layer.parameters():
            param.requires_grad = False
            

class BERTClassifier(MyBertModule):
    
    def __init__(self, bert_model_name, num_classes):
        super().__init__(bert_model_name)
        self.dropout = nn.Dropout(0.1)
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


class BERTRegressor(MyBertModule):
   
   def __init__(self, bert_model_name):
       super().__init__(bert_model_name)
       D_in, D_out = self.bert.config.hidden_size, 1
       self.ff_layer: nn.Module = nn.Sequential(
           nn.Dropout(0.2),
           nn.Linear(D_in, 20),
           nn.LeakyReLU(),
           nn.Dropout(0.2),
           nn.Linear(20, 20),
           nn.LeakyReLU(),
           nn.Dropout(0.2),
           nn.Linear(20, D_out)
           )

   def forward(self, input_ids, attention_masks):
       output = self.bert(input_ids, attention_mask=attention_masks)
       cls_tokens = output.last_hidden_state[:,0,:]
       out = self.ff_layer(cls_tokens)
       return out


def train_one_epoch(model: nn.Module, train_dataloader, device, loss_function, clip_value, optimizer: Optimizer, scheduler: LambdaLR, t0):

    epoch_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
            
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        inputs = (batch_inputs, batch_masks)
        
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs".
        model.zero_grad()
        
        outputs = model(*inputs)     
            
        batch_loss = loss_function(outputs.squeeze(), 
                                   batch_labels.squeeze())
        epoch_loss += batch_loss.item()
        
        batch_loss.backward() # Calculate gradients
        
        # Clip the norm of the gradients.
        # This is to help prevent the "exploding gradients" problem.
        clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step() # Update parameters
        
        scheduler.step() # Updapte learning rate
        
        if step % 1000 == 999:
            last_loss = epoch_loss / (step+1) 
            print('batch {} loss: {}'.format(step + 1, last_loss))
            
    return epoch_loss


def train(model: nn.Module, 
          optimizer: Optimizer, 
          scheduler: LambdaLR, 
          loss_function, epochs, 
          train_dataloader: DataLoader, 
          validation_dataloader, 
          device, 
          clip_value = 2,
          tracking_metrics: dict = dict()):
    
    training_stats = []
    t0 = time.time()
    
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, train_dataloader, device, loss_function, clip_value, optimizer, scheduler, t0)

        avg_train_loss = epoch_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        val_metrics: dict
        avg_val_loss, val_metrics = evaluate(model, 
                                             loss_function, 
                                             validation_dataloader, 
                                             device, 
                                             tracking_metrics)
        
        epoch_dict = {
                'epoch': epoch + 1,
                'Validation Loss': avg_val_loss,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
            }
        for name in val_metrics:
            epoch_dict["Valid." + name] = val_metrics[name]
        
        training_stats.append(epoch_dict)

        print("")
        print(pd.Series(epoch_dict))
        
    return model, training_stats
   

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