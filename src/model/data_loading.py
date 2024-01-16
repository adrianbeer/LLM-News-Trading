import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from src.config import config, MODEL_CONFIG
from src.model.neural_network import embed_inputs
from typing import List


def get_data_loader_from_dataset(dataset: pd. DataFrame, 
                                 split: str, 
                                 tokenizer, 
                                 batch_size: int, 
                                 data_loader_kwargs: dict = dict(),
                                 text_col: str = None,
                                 label_col: str = None):
    texts, labels = get_text_and_labels(dat=dataset, 
                                        split=split, 
                                        text_col=text_col, 
                                        label_col=label_col)
    inputs, masks = embed_inputs(texts, 
                                 tokenizer)
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


def get_text_and_labels(dat: pd.DataFrame, 
                        split: str = None,
                        text_col: str = None,
                        label_col: str = None) -> tuple[List, List]:
    if not text_col:
        text_col = MODEL_CONFIG.input_col_name
    if not label_col:
        label_col = MODEL_CONFIG.target_col_name
    if split:
        dat = dat.loc[dat["split"] == split, :] 
    texts = dat.loc[:, text_col].tolist()
    labels = dat.loc[:, label_col].tolist()
    return texts, labels
