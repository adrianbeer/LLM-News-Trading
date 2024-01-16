import yaml
from dotmap import DotMap
import os 
import logging
from src.model.neural_network import BERTClassifier, BERTRegressor
import torch.nn as nn
from src.model.splits import Splitter, DateSplitter, RatioSplitter

# logging
logging.basicConfig(filename='logs/logs.log', level=logging.INFO)

# Default location/file
config_path = "src/config.yaml"
try:
    config_path = os.environ["TRADING_BOT_CONFIG_PATH"]
except KeyError:
    pass
config = DotMap(yaml.safe_load(open(config_path)), _dynamic=False)


class MODEL_CONFIG:
    splitter: Splitter = DateSplitter(test_cutoff_date = "2021-01-01", 
                                      val_cutoff_date = "2022-01-01")
    BertClass: nn.Module = BERTClassifier
    transformer_hugface_id: str = 'yiyanghkust/finbert-fls'
    loss_function = nn.CrossEntropyLoss
    input_col_name: str =  "parsed_body"
    target_col_name: str = "z_score_class"
    
    
  
