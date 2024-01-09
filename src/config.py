import yaml
from dotmap import DotMap
import os 
import logging

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