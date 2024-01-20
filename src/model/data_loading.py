import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from src.config import config

def get_data_loader_from_dataset(dataset: pd. DataFrame, 
                                 split: str, 
                                 batch_size: int, 
                                 label_col: str,
                                 data_loader_kwargs: dict = dict(),
                                 ):
    if split:
        indices = dataset.loc[dataset["split"] == split, :].index
    
    input_ids: pd.DataFrame = pd.read_parquet(config.data.benzinga.input_ids)
    masks: pd.DataFrame = pd.read_parquet(config.data.benzinga.masks)
    labels = dataset[label_col]
    print(f"{dataset.index.name=}")

    tensors = []
    for item in input_ids, masks, labels:
        x = item.loc[indices]
        x = torch.from_numpy(x.to_numpy())
        tensors.append(x)
    
    dataloader = create_dataloader(tensors=tensors, 
                                   batch_size=batch_size, 
                                   data_loader_kwargs=dict(shuffle=False))
    return dataloader



def create_dataloader(tensors: List[Tensor], 
                      batch_size: int, 
                      data_loader_kwargs: dict = dict()) -> DataLoader:
    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            **data_loader_kwargs)
    return dataloader
