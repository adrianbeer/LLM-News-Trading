import pandas as pd
from src.config import config, MODEL_CONFIG

dat = pd.read_parquet(config.data.learning_dataset)
print(dat.describe())
dat.describe().to_csv("data/learn_dat_describe.csv")