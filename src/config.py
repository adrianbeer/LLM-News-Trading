import yaml
from dotmap import DotMap
import os 
from src.model.neural_network import BERTClassifier
import torch.nn as nn
from src.model.splits import Splitter, RatioSplitter
from dataclasses import dataclass

# Default location/file
config_path = "src/config.yaml"
try:
    config_path = os.environ["TRADING_BOT_CONFIG_PATH"]
except KeyError:
    pass
config = DotMap(yaml.safe_load(open(config_path)), _dynamic=False)


@dataclass(frozen=True)
class ModelConfig:
    task: str
    splitter: Splitter
    neural_net: nn.Module
    pretrained_network: str
    tokenizer: str
    input_col_name: str
    target_col_name: str
    input_params_path: str
    output_params_path: str
    
    
MODEL_CONFIG: ModelConfig = ModelConfig(
    task = "Classification", # or "Regression"
    splitter = RatioSplitter(0.7, 0.2),
    neural_net = BERTClassifier,
    pretrained_network = 'data/models/networks/finbert_tone',
    tokenizer = "data/models/tokenizers/finbert-tone",
    input_col_name =  "parsed_body",
    target_col_name = "z_score_class",
    input_params_path = "data/model_state_dict.pt",
    output_params_path = "data/model_state_dict.pt"
)

    
  
