import yaml
from dotmap import DotMap
import os 
import logging
from src.model.neural_network import BERTClassifier, BERTRegressor

# Default location/file
config_path = "src/config.yaml"

# Get path via environment variable iff possible 
try:
    config_path = os.environ["TRADING_BOT_CONFIG_PATH"]
except KeyError:
    pass

config = DotMap(yaml.safe_load(open(config_path)), _dynamic=False)

# logging
logging.basicConfig(filename='logs/logs.log', level=logging.INFO)

# ---
MODEL_CONFIG = {
    "data" : {
        "test_cutoff_date":"2021-01-01",
        "val_cutoff_date": "2022-01-01"
    },
    "BertClass": BERTRegressor,
    "input_col_name": "parsed_body",
    "target_col_name": "r_mkt_adj",
    "transformer_hugface_id": 'yiyanghkust/finbert-fls'
    
}
MODEL_CONFIG = DotMap(MODEL_CONFIG)

    
  
