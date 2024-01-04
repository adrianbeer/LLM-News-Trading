import datetime
import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dotmap import DotMap
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel

from src.evaluation import METRICS_FUNCTION_DICT

config = DotMap(yaml.safe_load(open("src/config.yaml")), _dynamic=False)

TRANSFORMER_HF_ID = 'yiyanghkust/finbert-fls'

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def create_dataloaders(inputs: Tensor, masks: Tensor, labels: List, batch_size: int) -> DataLoader:
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader

# class MyNNConfig(PretrainedConfig)

class MyBertModel(nn.Module):
   
   def __init__(self):
       super(MyBertModel, self).__init__()
       self.bert = BertModel.from_pretrained(TRANSFORMER_HF_ID)
       D_in, D_out = self.bert.config.hidden_size, 1
       self.regr = nn.Sequential(
           nn.Dropout(0.2),
           nn.Linear(D_in, D_out)
           )

   def forward(self, input_ids, attention_masks):
       output = self.bert(input_ids, attention_mask=attention_masks)
       out = self.regr(output.last_hidden_state[:,0,:])
       return out


def train(model: nn.Module, optimizer, scheduler, loss_function, epochs, train_dataloader, validation_dataloader, device, clip_value=2):
    
    training_stats = []
    t0 = time.time()

    for epoch in range(epochs):
        print(f'======== Epoch {epoch} / {epochs} ========')
        print('Training...')
        
        total_train_loss = 0
        # best_loss = 1e10
        model.train()

        for step, batch in enumerate(train_dataloader): 
            batch_inputs, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs".
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)           
            loss = loss_function(outputs.squeeze(), 
                             batch_labels.squeeze())
            total_train_loss += loss.item()
            # Calculate gradients
            loss.backward() 
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            clip_grad_norm(model.parameters(), clip_value)
            # Update parameters
            optimizer.step()
            # Updapte learning rate
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        val_loss: float
        val_metrics: dict
        val_loss, val_metrics = evaluate(model, loss_function, validation_dataloader, device)
        
        epoch_dict = {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': val_loss,
                'Training Time': training_time,
            }
        for name in val_metrics:
            epoch_dict["Valid." + name] = val_metrics[name]
        
        training_stats.append(epoch_dict)

        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epoch took: {:}".format(training_time))        
    return model, training_stats
   

@torch.no_grad
def evaluate(model: nn.Module, loss_function, validation_dataloader: DataLoader, device):
    model.eval()
    test_loss = []
    
    metrics_batched = dict([(name, []) for name in METRICS_FUNCTION_DICT])
    
    for batch in validation_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        outputs = model(batch_inputs, batch_masks)

        loss: torch.Tensor = loss_function(outputs, batch_labels.unsqueeze(1))
        outputs: np.ndarray = outputs.to('cpu').numpy()
        labels: np.ndarray = batch_labels.to('cpu').numpy()

        test_loss.append(loss.item())
        
        for metric_name in metrics_batched:
            metrics_batched[metric_name].append(METRICS_FUNCTION_DICT[metric_name](outputs, labels))
        
    test_loss = np.mean(test_loss)
    metrics = dict([(name, np.mean(metrics_batched[name])) for name in metrics_batched])
    return test_loss, metrics


def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = \
                                  tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, 
                            batch_masks).view(1,-1).tolist()[0]
    return np.array(output)


def embed_input(text, tokenizer):
    # Truncation = True as bert can only take inputs of max 512 tokens.
    # return_tensors = "pt" makes the funciton return PyTorch tensors
    # tokenizer.encode_plus specifically returns a dictionary of values instead of just a list of values
    encoding = tokenizer(
        text, 
        add_special_tokens = True, 
        truncation = True, 
        padding = "max_length", 
        max_length = 512,
        return_attention_mask = True, 
        return_tensors = "pt"
    )
    # input_ids: mapping the words to tokens
    # attention masks: idicates if index is word or padding
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    return input_ids, attention_masks

from functools import wraps
from time import time
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

@timing
def embed_inputs(texts: list, tokenizer) -> tuple[Tensor, Tensor]:
    input_ids = []
    attention_masks = []
    for text in texts:
        x, y = embed_input(text, tokenizer)
        input_ids.append(x)
        attention_masks.append(y)
    input_ids: Tensor = torch.cat(input_ids, dim=0)
    attention_masks: Tensor = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks
    

class WeightedSquaredLoss(nn.Module):
    def __init__(self, gamma):
        super(WeightedSquaredLoss, self).__init__()
        self.gamma = gamma

    def forward(self, output, target):
        flat_output = torch.flatten(output)
        flat_target = torch.flatten(target)
        N = len(output)
        loss = torch.dot(torch.pow(torch.add(torch.abs(flat_output), 1), self.gamma), torch.square(flat_output - flat_target)) / N
        return loss
    
def get_text_and_labels(dataset: pd.DataFrame, section: str):
    input_col_name = config.model.input_col_name
    target_col_name = config.model.target_col_name
    dat = dataset.loc[dataset.section == section, :]
    texts = dat.loc[:, input_col_name].tolist()
    labels = dat.loc[:, target_col_name].tolist()
    return texts, labels



