import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from src.config import config
from torch.utils.data import Dataset
import lightning as pl
import numpy as np

class CustomDataset(Dataset):
    
    def __init__(self, 
                 news_data, 
                 input_ids, 
                 masks, 
                 stage, 
                 target_col_name, 
                 news_data_idx=None):
        self.stage = stage
        self.news_data = news_data
        # For the test modules where only some indices are selected for unit testing
        if news_data_idx: 
            self.news_data = self.news_data.loc[news_data_idx, :]
            
        if self.stage:
            self.news_data = self.news_data.loc[self.news_data.split == self.stage, :]
        
        self.sample_weights = self.news_data.loc[:, 'sample_weights']
        self.is_overnight_news = self.news_data.loc[:, 'is_overnight_news']
        self.news_data = self.news_data.loc[:, target_col_name]
        
        self.input_ids = input_ids
        self.masks = masks
        
        self.input_ids = self.input_ids.loc[self.news_data.index, :]
        self.masks = self.masks.loc[self.news_data.index, :]
        
        assert (self.news_data.index == self.input_ids.index).all()
        assert (self.news_data.index == self.masks.index).all()
        
    def get_baseline_mae(self):
        return np.abs((self.news_data - self.news_data.median())).mean()

    def get_baseline_mse(self):
        return ((self.news_data - self.news_data.mean())**2).mean()

    def get_class_distribution(self):
        class_distribution = (self.news_data.value_counts() / self.news_data.shape[0]).sort_index()
        return class_distribution
        
    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        news_data = torch.tensor(self.news_data.iloc[idx])
        sample_weights = torch.tensor(self.sample_weights.iloc[idx])
        input_ids = torch.from_numpy(self.input_ids.iloc[idx, :].values) 
        masks = torch.from_numpy(self.masks.iloc[idx, :].values) 
        #! Change this to something like misc_indicators
        is_overnight_news = torch.tensor(self.is_overnight_news.iloc[idx])
        
        sample = {'target': news_data, 
                  'input_id': input_ids, 
                  'mask': masks,
                  'sample_weights': sample_weights,
                  'is_overnight_news': is_overnight_news}
        
        return sample


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, 
                 news_data_path: str, 
                 input_ids_path: str, 
                 masks_path: str, 
                 batch_size: int, 
                 target_col_name: str, 
                 news_data_idx: int = None):
        super().__init__()
        self.news_data = pd.read_parquet(news_data_path, columns=[target_col_name, 'sample_weights', 'split'])
        
        self.input_ids = pd.read_parquet(input_ids_path)
        self.masks = pd.read_parquet(masks_path)
        
        self.batch_size = batch_size
        self.target_col_name = target_col_name
        self.news_data_idx = news_data_idx

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = CustomDataset(news_data=self.news_data, 
                                                input_ids=self.input_ids, 
                                                masks=self.masks,
                                                stage="training",
                                                target_col_name=self.target_col_name,
                                                news_data_idx=self.news_data_idx)
            self.val_dataset = CustomDataset(news_data=self.news_data, 
                                                input_ids=self.input_ids, 
                                                masks=self.masks,
                                                stage="validation",
                                                target_col_name=self.target_col_name,
                                                news_data_idx=self.news_data_idx)
            print(
                f"baseline MAE (train): {self.train_dataset.get_baseline_mae()} \n"
                f"baseline MAE (val): {self.val_dataset.get_baseline_mae()} \n"
                f"baseline MSE (train): {self.train_dataset.get_baseline_mse()} \n"
                f"baseline MSE (val): {self.val_dataset.get_baseline_mse()}"
                )
        if stage == "test":
            self.test_dataset = CustomDataset(news_data=self.news_data, 
                                                input_ids=self.input_ids, 
                                                masks=self.masks,
                                                stage="testing",
                                                target_col_name=self.target_col_name,
                                                news_data_idx=self.news_data_idx)
        if stage == "predict":
            self.predict_dataset = CustomDataset(news_data=self.news_data, 
                                                input_ids=self.input_ids, 
                                                masks=self.masks,
                                                stage=None,
                                                target_col_name=self.target_col_name,
                                                news_data_idx=self.news_data_idx)
        if stage not in ['fit', 'test', 'predict']:
            raise ValueError('Invalid stage.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          pin_memory=True,
                          num_workers=3)


def get_data_loader_from_dataset(dataset: pd. DataFrame, 
                                 split: str, 
                                 batch_size: int, 
                                 label_col: str,
                                 data_loader_kwargs: dict = dict(),
                                 ):
    if split:
        indices = dataset.loc[dataset["split"] == split, :].index
    
    input_ids: pd.DataFrame = pd.read_parquet(config.data.news.input_ids)
    masks: pd.DataFrame = pd.read_parquet(config.data.news.masks)
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


class MLMDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        self.ids = pd.read_parquet("data/news/input_ids.parquet")
        self.masks = pd.read_parquet("data/news/masks.parquet")
        assert (self.ids.index == self.masks.index).all()
        self.ids = self.ids.values
        self.masks = self.masks.values
        
        N = len(self.ids)
        cutoff = int(N*0.1)
        if evaluate:
            self.ids = self.ids[:cutoff, :]
            self.masks = self.masks[:cutoff, :]
        else:
            self.ids = self.ids[cutoff:, :]
            self.masks = self.masks[cutoff:, :]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return { 
                "input_ids": self.ids[i],
                "attention_mask": self.masks[i]
                }