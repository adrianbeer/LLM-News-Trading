import torch
from src.config import config
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from lightning import Trainer

from src.model.data_loading import CustomDataModule
from src.model.regr_transformer import NNRegressor
from src.model.bert_classifier import BERTClassifier

dataset = pd.read_parquet(config.data.learning_dataset)

# ---- Classification Model ---- 
BATCH_SIZE = 64
target_col_name = 'z_score_class'
model_id = '7zdo79pv'
model_step = 'epoch=3-step=171548'

torch.cuda.empty_cache()
dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                      input_ids_path=config.data.news.input_ids, 
                      masks_path=config.data.news.masks, 
                      batch_size=64,
                      target_col_name=target_col_name)

model = BERTClassifier.load_from_checkpoint(f"data/ckpts/{model_id}/{model_step}.ckpt", num_classes=3)
dm.setup('predict')

# disable randomness, dropout, etc...
model.eval()
    
data_loader = dm.predict_dataloader()
trainer = Trainer()
logits = torch.cat(trainer.predict(model, data_loader), dim=0)

probs = logits.softmax(dim=1)
max_probs = np.apply_along_axis(np.max, axis=1, arr=probs)
class_preds = np.apply_along_axis(np.argmax, axis=1, arr=probs)
dataset.loc[:, "max_probs"] = max_probs
dataset.loc[:, "class_preds"] = class_preds

dataset.to_parquet(f"{config.data.predictions.regression_dir}/{model_id}.parquet")