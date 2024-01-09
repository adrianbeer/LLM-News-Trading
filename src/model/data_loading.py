import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from src.config import config
from src.model.neural_network import embed_inputs
from typing import List


def get_data_loader_from_dataset(dataset:pd. DataFrame, split, tokenizer, batch_size):
    texts, labels = get_text_and_labels(dataset, split)
    inputs, masks = embed_inputs(texts, tokenizer)
    dataloader: DataLoader = create_dataloaders(inputs, masks, labels, batch_size)
    return dataloader


def create_dataloaders(inputs: Tensor, masks: Tensor, labels: List, batch_size: int) -> DataLoader:
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    return dataloader


def get_text_and_labels(dataset: pd.DataFrame, split: str):
    input_col_name = config.model.input_col_name
    target_col_name = config.model.target_col_name
    dat = dataset.loc[dataset["split"] == split, :]
    texts = dat.loc[:, input_col_name].tolist()
    labels = dat.loc[:, target_col_name].tolist()
    return texts, labels