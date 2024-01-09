
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dotmap import DotMap
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from src.model.data_loading import create_dataloaders, get_text_and_labels, get_data_loader_from_dataset

from src.model.neural_network import (
    TRANSFORMER_HF_ID,
    MyBertModel,
    WeightedSquaredLoss,
    embed_inputs,
    train,
)

config = DotMap(yaml.safe_load(open("src/config.yaml")), _dynamic=False)
input_col_name = config.model.input_col_name
target_col_name = config.model.target_col_name

# Settings
FROM_SCRATCH = True
batch_size = 4
tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_HF_ID)


# Download dataset
dataset = pd.read_parquet(config.data.merged, columns=[input_col_name, target_col_name, "section"])

validation_dataloader = get_data_loader_from_dataset(dataset=dataset, 
                                                split="validation", 
                                                tokenizer=tokenizer, 
                                                batch_size=batch_size,
                                                data_loader_kwargs={})

train_dataloader = get_data_loader_from_dataset(dataset=dataset, 
                                                split="training", 
                                                tokenizer=tokenizer, 
                                                batch_size=batch_size,
                                                data_loader_kwargs={})


model: nn.Module = MyBertModel()
if not FROM_SCRATCH: 
    model.load_state_dict(torch.load("data/model")) # Use latest iteration of the model for training
    
# .compile currently isn't supported for Windows
model = torch.compile(model)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
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
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)
    
    loss_function = nn.MSELoss()
    
    # Training
    model, training_stats = train(model, 
                                  optimizer, 
                                  scheduler, 
                                  loss_function, 
                                  epochs, 
                                  train_dataloader, 
                                  validation_dataloader, 
                                  device, 
                                  clip_value=2)

    df_stats = pd.DataFrame(data=training_stats)
    print(df_stats)

    # Store Model
    torch.save(model.state_dict(), "data/model")



