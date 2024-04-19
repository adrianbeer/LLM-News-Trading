import yaml
from dotmap import DotMap
import os 
from src.model.bert_classifier import BERTClassifier
from src.model.regr_transformer import NNRegressor
import torch.nn as nn
from src.model.splits import Splitter, RatioSplitter, DateSplitter
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
    neural_net: nn.Module
    base_model: str
    masks: str
    input_ids: str
    target_col_name: str

@dataclass(frozen=True)
class PreprocessingConfig: 
    splitter: Splitter
    tokenizer: str
    input_col_name: str
    target_col_name: str

PREP_CONFIG = PreprocessingConfig(
    splitter = DateSplitter(val_cutoff_date="2022-01-01", test_cutoff_date="2023-01-01", time_column="est_entry_time"),
    tokenizer = "data/models/newstokenizer",
    input_col_name =  "parsed_body",
    target_col_name = "z_score",
)  

ClassificationConfig: ModelConfig = ModelConfig(
    task = "Classification",
    base_model = 'data/models/roberta_mlm/checkpoint-900000', 
    neural_net = BERTClassifier,
    masks = config.data.news.masks, 
    input_ids = config.data.news.input_ids,
    target_col_name = "z_score_2_class",
)

RegressorConfig: ModelConfig = ModelConfig(
    task = "Regression",
    base_model = 'data/models/roberta_mlm/checkpoint-900000', 
    neural_net = NNRegressor,
    masks = config.data.news.masks, 
    input_ids = config.data.news.input_ids,
    target_col_name = "z_score",
)
