import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List


def get_data_loader_from_dataset(dataset: pd. DataFrame, 
                                 encoding_matrix_path: str,
                                 split: str, 
                                 batch_size: int, 
                                 label_col: str,
                                 data_loader_kwargs: dict = dict()):
    if split:
        indices = dataset.loc[dataset["split"] == split, :].index
    
    dat_indices = dataset.index
    enc_indices, input_ids, masks = get_encoding(encoding_matrix_path)

    labels: pd.Series = dataset[label_col]
    dataloader: DataLoader = create_dataloader([input_ids, 
                                                masks, 
                                                labels], 
                                                batch_size, 
                                                data_loader_kwargs)
    return dataloader



def create_dataloader(tensors: List[Tensor], 
                      batch_size: int, 
                      data_loader_kwargs: dict = dict()) -> DataLoader:
    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            **data_loader_kwargs)
    return dataloader
