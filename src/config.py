import yaml
from dotmap import DotMap
import os 
import logging
from src.model.neural_network import BERTClassifier, BERTRegressor
import torch.nn as nn
from src.model.splits import Splitter, DateSplitter, RatioSplitter
from dataclasses import dataclass
from typing import Any

# logging
logging.basicConfig(filename='logs/logs.log', level=logging.INFO)

# Default location/file
config_path = "src/config.yaml"
try:
    config_path = os.environ["TRADING_BOT_CONFIG_PATH"]
except KeyError:
    pass
config = DotMap(yaml.safe_load(open(config_path)), _dynamic=False)


@dataclass(frozen=True)
class ModelConfig:
    splitter: Splitter
    BertClass: nn.Module
    transformer_hugface_id: str
    loss_function: Any
    input_col_name: str
    target_col_name: str
    
    
MODEL_CONFIG: ModelConfig = ModelConfig(
    splitter = RatioSplitter(0.7, 0.2),
    BertClass = BERTClassifier,
    transformer_hugface_id = 'yiyanghkust/finbert-fls',
    loss_function = nn.CrossEntropyLoss,
    input_col_name =  "parsed_body",
    target_col_name = "z_score_class",
)

    
  
