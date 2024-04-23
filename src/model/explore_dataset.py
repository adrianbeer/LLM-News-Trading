import pandas as pd
from src.model.data_loading import CustomDataModule
from src.config import config, ClassificationConfig

from transformers import AutoModel
import torch.nn as nn

dm = CustomDataModule(news_data_path=config.data.learning_dataset, 
                      input_ids_path=ClassificationConfig.input_ids, 
                      masks_path=ClassificationConfig.masks, 
                      batch_size=4,
                      target_col_name=ClassificationConfig.target_col_name)

dm.setup('fit')
data_loader = dm.train_dataloader()

first_batch = next(iter(data_loader))
for key in first_batch:
    print(key)
    print(first_batch[key])
    
network: nn.Module = AutoModel.from_pretrained(ClassificationConfig.base_model)
outputs = network(first_batch['input_id'], first_batch['mask'])
print(f"{outputs=}")
# -- 
