import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


from typing import List

from src.model.neural_network import config


def create_dataloaders(inputs: Tensor, masks: Tensor, labels: List, batch_size: int) -> DataLoader:
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    return dataloader


def get_text_and_labels(dataset: pd.DataFrame, section: str):
    input_col_name = config.model.input_col_name
    target_col_name = config.model.target_col_name
    dat = dataset.loc[dataset.section == section, :]
    texts = dat.loc[:, input_col_name].tolist()
    labels = dat.loc[:, target_col_name].tolist()
    return texts, labels