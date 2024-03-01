import pandas as pd
from src.config import config, MODEL_CONFIG

dat = pd.read_parquet(config.data.learning_dataset)
print(dat.describe())

print(f"{dat.max()=}")
print(f"{dat.min()=}")