from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertModel
from torch.nn.utils.clip_grad import clip_grad_norm
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from util import create_dataloaders, MyBertModel, train, TRANSFORMER_HF_ID, predict
import pickle

# Download dataset
dataset = pd.read_pickle("data/dataset.pkl")

### Input formatting
texts = dataset.body.tolist()
labels = dataset.IntradayReturn.tolist()
batch_size = 2
seed = 420
test_size = 0.1

tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_HF_ID)

input_ids = []
attention_masks = []
for text in texts:
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
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])
    
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=test_size, random_state=seed)
train_masks, test_masks, _, _ = train_test_split(attention_masks, labels, test_size=test_size, random_state=seed)

train_dataloader = create_dataloaders(train_inputs, train_masks, 
                                      train_labels, batch_size)
validation_dataloader = create_dataloaders(test_inputs, test_masks, 
                                     test_labels, batch_size)


model = MyBertModel()

if __name__ == "__main__":
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    # Optimizer, scheduler and loss function
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    epochs = 5
    total_steps = len(text) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=total_steps)
    loss_function = nn.MSELoss()

    # Training
    model, training_stats = train(model, optimizer, scheduler, loss_function, epochs, 
                train_dataloader, validation_dataloader, device, clip_value=2)

    df_stats = pd.DataFrame(data=training_stats)
    print(df_stats)

    # Store Model
    torch.save(model.state_dict(), "data/model")



