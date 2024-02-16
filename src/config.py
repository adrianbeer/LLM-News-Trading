import yaml
from dotmap import DotMap
import os 
from src.model.bert_classifier import BERTClassifier
from src.model.bert_regressor import BERTRegressor
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
    
    
ClassificationConfig: ModelConfig = ModelConfig(
    task = "Classification", # or "Regression"
    splitter = RatioSplitter(0.75, 0.15),
    pretrained_network = 'data/models/networks/finbert_tone',
    neural_net = BERTClassifier,
    tokenizer = "data/models/tokenizers/finbert-tone",
    input_col_name =  "parsed_body",
    target_col_name = "z_score_class",
)

    
RegressorConfig: ModelConfig = ModelConfig(
    task = "Regression",
    splitter = RatioSplitter(0.75, 0.15),
    pretrained_network = 'data/models/networks/finbert_tone',
    neural_net = BERTRegressor,
    tokenizer = "data/models/tokenizers/finbert-tone",
    input_col_name =  "parsed_body",
    target_col_name = "z_score",
)

MODEL_CONFIG = RegressorConfig
