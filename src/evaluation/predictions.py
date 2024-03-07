import torch
from src.model.bert_regressor import BertRegressor
from src.config import config, MODEL_CONFIG
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model.data_loading import CustomDataModule

dataset = pd.read_parquet(config.data.learning_dataset)
target_col_name = MODEL_CONFIG.target_col_name

torch.cuda.empty_cache()
dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                      input_ids_path=config.data.news.input_ids, 
                      masks_path=config.data.news.masks, 
                      batch_size=64,
                      target_col_name=target_col_name)


model = BertRegressor.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc...
model.eval()
with torch.no_grad():
    preds = model(dm.predict_dataloader())
    
dataset.loc[:, "prediction"] = preds
dataset.to_parquet(config.data.learning_dataset)
