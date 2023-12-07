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
from src.model.util import create_dataloaders, MyBertModel, train, TRANSFORMER_HF_ID, embed_input, embed_inputs, WeightedSquaredLoss
import pickle
from transformers import BertTokenizer
from src.model.util import embed_inputs, create_dataloaders, TRANSFORMER_HF_ID
from src.config import TARGET_COL_NAME

FROM_SCRATCH = True
batch_size = 4
loss_confidence_parameter = 1 # Je höher, desto größer ist die Aussagekraft einer hohen Prognose

# Download dataset
dataset = pd.read_pickle("data/dataset.pkl")
(train_idx, test_idx) = pd.read_pickle("data/dataset_train_test_idx.pkl")

train_dat = dataset.loc[train_idx, :]
test_dat = dataset.loc[test_idx, :]
print(f"train_dat size: {train_dat.shape[0]}")

tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_HF_ID)

train_texts = train_dat.body.tolist()
test_texts = test_dat.body.tolist()
test_labels = test_dat.loc[:, TARGET_COL_NAME].tolist()
train_labels = train_dat.loc[:, TARGET_COL_NAME].tolist()

train_inputs, train_masks = embed_inputs(train_texts, tokenizer)
test_inputs, test_masks = embed_inputs(test_texts, tokenizer)

train_dataloader = create_dataloaders(train_inputs, train_masks, 
                                      train_labels, batch_size)
validation_dataloader = create_dataloaders(test_inputs, test_masks, 
                                     test_labels, batch_size)


model = MyBertModel()
if not FROM_SCRATCH: model.load_state_dict(torch.load("data/model")) # Use latest iteration of the model for training

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
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=total_steps)
    loss_function = WeightedSquaredLoss(gamma=loss_confidence_parameter)

    # Training
    model, training_stats = train(model, optimizer, scheduler, loss_function, epochs, 
                train_dataloader, validation_dataloader, device, clip_value=2)

    df_stats = pd.DataFrame(data=training_stats)
    print(df_stats)

    # Store Model
    torch.save(model.state_dict(), "data/model")



