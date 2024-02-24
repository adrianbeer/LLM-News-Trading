import yaml
from dotmap import DotMap
import os 
from src.model.bert_classifier import BERTClassifier
from src.model.bert_regressor import BERTRegressor
import torch.nn as nn
from src.model.splits import Splitter, RatioSplitter
from dataclasses import dataclass

# Default location/file
data_config_path = "src/data_config.yaml"
try:
    data_config_path = os.environ["TRADING_BOT_CONFIG_PATH"]
except KeyError:
    pass
config = DotMap(yaml.safe_load(open(data_config_path)), _dynamic=False)


@dataclass(frozen=True)
class ModelConfig:
    task: str
    splitter: Splitter
    neural_net: nn.Module
    pretrained_network: str
    input_col_name: str
    target_col_name: str

@dataclass(frozen=True)
class PreprocessingConfig: 
    splitter: Splitter
    tokenizer: str
    input_col_name: str
    target_col_name: str

PREP_CONFIG = PreprocessingConfig(
    splitter = RatioSplitter(0.75, 0.15),
    tokenizer = "data/models/tokenizers/finbert-tone",
    input_col_name =  "parsed_body",
    target_col_name = "z_score",
)  
    
ClassificationConfig: ModelConfig = ModelConfig(
    task = "Classification", # or "Regression"
    splitter = RatioSplitter(0.75, 0.15),
    pretrained_network = 'data/models/networks/finbert_tone',
    neural_net = BERTClassifier,
    input_col_name =  "parsed_body",
    target_col_name = "z_score_class",
)

RegressorConfig: ModelConfig = ModelConfig(
    task = "Regression",
    splitter = RatioSplitter(0.75, 0.15),
    pretrained_network = 'data/models/networks/finbert_tone',
    neural_net = BERTRegressor,
    input_col_name =  "parsed_body",
    target_col_name = "z_score",
)

MODEL_CONFIG = RegressorConfig
