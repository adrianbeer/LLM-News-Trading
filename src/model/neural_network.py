import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotmap import DotMap
from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import BertModel
from src.utils.time import timing, format_time
from src.evaluation.metrics import METRICS_FUNCTION_DICT
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from src.config import config

TRANSFORMER_HF_ID = 'yiyanghkust/finbert-fls'


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


def train(model: nn.Module, optimizer, scheduler, loss_function, epochs, train_dataloader: DataLoader, validation_dataloader, device, clip_value=2, N_train=0):
    
    training_stats = []
    t0 = time.time()
    epoch_time_is_estimated = False
    
    for epoch in range(epochs):
        print(f'======== Epoch {epoch} / {epochs} ========')
        print(f'Training... at {format_time(t0)}')
        
        total_train_loss = 0
        # best_loss = 1e10
        model.train()

        for step, batch in enumerate(train_dataloader):
            
            if (step*train_dataloader.batch_size >= 10_000) and not epoch_time_is_estimated: 
                print(f"One epoch takes take approx. {N_train / 10_000 * (time.time() - t0)/(60*60)} hours")
                epoch_time_is_estimated = True
                
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
            clip_grad_norm_(model.parameters(), clip_value)
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


@timing
def embed_inputs(texts: list, tokenizer) -> tuple[Tensor, Tensor]:
    input_ids = []
    attention_masks = []
    
    pool_obj = ThreadPoolExecutor(max_workers=os.cpu_count())
    ans = pool_obj.map(partial(embed_input, tokenizer=tokenizer), texts)
    input_ids, attention_masks = list(zip(*ans))

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
    




