from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, PretrainedConfig
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertModel
from torch.nn.utils.clip_grad import clip_grad_norm
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import datetime
import numpy as np

TRANSFORMER_HF_ID = 'yiyanghkust/finbert-tone'

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def create_dataloaders(inputs, masks, labels, batch_size):
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
       self.regr = nn.Sequential(nn.Linear(D_in, D_out))

   def forward(self, input_ids, attention_masks):
       output = self.bert(input_ids, attention_mask=attention_masks)
       out = self.regr(output.last_hidden_state[:,0,:])
       return out


def train(model, optimizer, scheduler, loss_function, epochs, train_dataloader, validation_dataloader, device, clip_value=2):
    
    training_stats = []
    t0 = time.time()

    for epoch in range(epochs):
        print(f'======== Epoch {epoch} / {epochs} ========')
        print('Training...')
        
        total_train_loss = 0
        best_loss = 1e10
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

        val_loss, val_mae = evaluate(model, loss_function, validation_dataloader, device)
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss':np.mean(val_loss),
                'Valid. MAE.': np.mean(val_mae),
                'Training Time': training_time,
                # 'Validation Time': validation_time
            }
        )

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))        
    return model, training_stats
   
def evaluate(model, loss_function, validation_dataloader, device):
    model.eval()
    test_loss, test_mae = [], []
    for batch in validation_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels.unsqueeze(1))

        outputs = outputs.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()

        test_loss.append(loss.item())
        mae = mae_score(outputs, label_ids)
        test_mae.append(mae.item())
    return test_loss, test_mae


def mae_score(outputs, labels):
    return np.mean(np.abs(outputs - labels))

def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = \
                                  tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, 
                            batch_masks).view(1,-1).tolist()[0]
    return output

