import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from src.config import config
from torch.utils.data import Dataset
import lightning as pl


class CustomDataset(Dataset):
    
    def __init__(self, news_data_path, input_ids_path, masks_path, stage, target_col_name):
        self.news_data = pd.read_parquet(news_data_path)
        self.stage = stage
        
        self.news_data = self.news_data.loc[self.news_data.split == stage, target_col_name]
        self.class_weights = (self.news_data.shape[0] / self.news_data.value_counts()).values
        
        self.input_ids = pd.read_parquet(input_ids_path)
        self.masks = pd.read_parquet(masks_path)
        
        self.input_ids = self.input_ids.loc[self.news_data.index, :]
        self.masks = self.masks.loc[self.news_data.index, :]
        
        assert (self.news_data.index == self.input_ids.index).all()
        assert (self.news_data.index == self.masks.index).all()
        
    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        news_data = torch.tensor(self.news_data.iloc[idx])
        input_ids = torch.from_numpy(self.input_ids.iloc[idx, :].values) 
        masks = torch.from_numpy(self.masks.iloc[idx, :].values) 
        
        sample = {'target': news_data, 
                  'input_id': input_ids, 
                  "mask": masks}
        
        return sample

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, news_data_path, input_ids_path, masks_path, batch_size, target_col_name):
        super().__init__()
        self.news_data_path = news_data_path
        self.input_ids_path = input_ids_path 
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.target_col_name = target_col_name

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = CustomDataset(news_data_path=self.news_data_path, 
                                                input_ids_path=self.input_ids_path, 
                                                masks_path=self.masks_path,
                                                stage="training",
                                                target_col_name=self.target_col_name)
            self.val_dataset = CustomDataset(news_data_path=self.news_data_path, 
                                                input_ids_path=self.input_ids_path, 
                                                masks_path=self.masks_path,
                                                stage="validation",
                                                target_col_name=self.target_col_name)
        if stage == "test":
            self.test_dataset = CustomDataset(news_data_path=self.news_data_path, 
                                                input_ids_path=self.input_ids_path, 
                                                masks_path=self.masks_path,
                                                stage="testing",
                                                target_col_name=self.target_col_name)
        if stage == "predict":
            pass

    def get_weights(self):
        return self.train_dataset.class_weights

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        pass


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
