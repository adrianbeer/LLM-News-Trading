import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List


def get_data_loader_from_dataset(dataset: pd. DataFrame, 
                                 split: str, 
                                 batch_size: int, 
                                 label_col: str,
                                 data_loader_kwargs: dict = dict()):
    if split:
        dataset = dataset.loc[dataset["split"] == split, :] 
    inputs: pd.Series = dataset["input_id"]
    masks: pd.Series = dataset["mask"]
    labels: pd.Series = dataset[label_col]
    dataloader: DataLoader = create_dataloaders(inputs, 
                                                masks, 
                                                labels, 
                                                batch_size, 
                                                data_loader_kwargs)
    return dataloader


def create_dataloaders(inputs: Tensor, 
                       masks: Tensor, 
                       labels: List, 
                       batch_size: int, 
                       data_loader_kwargs: dict = dict()) -> DataLoader:
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, 
                            mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            **data_loader_kwargs)
    return dataloader
