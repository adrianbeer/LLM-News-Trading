
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dotmap import DotMap
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from time import time

config = DotMap(yaml.safe_load(open("src/config.yaml")), _dynamic=False)

from src.model.util import (
    TRANSFORMER_HF_ID,
    MyBertModel,
    WeightedSquaredLoss,
    create_dataloaders,
    embed_inputs,
    get_text_and_labels,
    train,
)

FROM_SCRATCH = True
batch_size = 2**12
loss_confidence_parameter = 1 # Je höher, desto größer ist die Aussagekraft einer hohen Prognose

input_col_name = config.model.input_col_name
target_col_name = config.model.target_col_name

tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_HF_ID)

# Download dataset
dataset = pd.read_parquet(config.data.merged, columns=[input_col_name, target_col_name, "section"])

train_texts, train_labels = get_text_and_labels(dataset, "training")
test_texts, test_labels = get_text_and_labels(dataset, "validation")
print(f"train_dat size: {len(train_texts)}")

print(time())
train_inputs, train_masks = embed_inputs(train_texts, tokenizer)
test_inputs, test_masks = embed_inputs(test_texts, tokenizer)

print(time())
train_dataloader = create_dataloaders(train_inputs, train_masks, 
                                      train_labels, batch_size)
print(time())
validation_dataloader = create_dataloaders(test_inputs, test_masks, 
                                     test_labels, batch_size)
print(time())

model: nn.Module = MyBertModel()
if not FROM_SCRATCH: 
    model.load_state_dict(torch.load("data/model")) # Use latest iteration of the model for training

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

    epochs = 1
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



