from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
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


class MyBertModel(nn.Module):
   
   def __init__(self):
       super(MyBertModel, self).__init__()
       self.bert = BertModel.from_pretrained('yiyanghkust/finbert-tone')
       D_in, D_out = self.bert.config.hidden_size, 1
       self.regr = nn.Sequential(nn.Linear(D_in, D_out))

   def forward(self, input_ids, attention_masks):
       output = self.bert(input_ids, attention_mask=attention_masks)
       out = self.regr(output.last_hidden_state[:,0,:])
       return out


def train(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, device, clip_value=2):
    for epoch in range(epochs):
        print(f'======== Epoch {epoch} / {epochs} ========')
        print('Training...')
        t0 = time.time()
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
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))        
    return model
   
def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss, test_r2 = [], []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())
    return test_loss, test_r2


def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

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