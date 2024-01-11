import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from src.config import config
from src.model.neural_network import embed_inputs
from typing import List


def get_data_loader_from_dataset(dataset: pd. DataFrame, 
                                 split: str, 
                                 tokenizer, 
                                 batch_size: int, 
                                 data_loader_kwargs: dict = dict()):
    texts, labels = get_text_and_labels(dataset, split)
    inputs, masks = embed_inputs(texts, tokenizer)
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


def get_text_and_labels(dat: pd.DataFrame, split: str = None) -> tuple[List, List]:
    input_col_name = config.model.input_col_name
    target_col_name = config.model.target_col_name
    if split:
        dat = dat.loc[dat["split"] == split, :] 
    texts = dat.loc[:, input_col_name].tolist()
    labels = dat.loc[:, target_col_name].tolist()
    return texts, labels