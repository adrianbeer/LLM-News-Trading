from src.config import config, MODEL_CONFIG
from os import listdir
from os.path import isfile, join
import numpy as np

def get_gcs_tickers():
  from google.cloud import storage
  storage_client = storage.Client()
  bucket = config.data.iqfeed.minute.raw.split("/")[2]
  prefix = "/".join(config.data.iqfeed.minute.raw.split("/")[3:]) + "/"
  bucket = storage_client.get_bucket(bucket)
  blobs = bucket.list_blobs(prefix=prefix)
  tickers = [subpath.name.split("/")[-1].split("_")[0] for subpath in blobs]
  tickers = [t for t in tickers if t != '']
  return tickers

def get_local_tickers(directory: str = None):
  if directory is None: 
    directory = config.data.iqfeed.minute.raw
  onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
  tickers = [x.split("_")[0] for x in onlyfiles]
  return tickers

def get_tickers(directory=None):
  if config.environment == "colab":
    tickers = get_gcs_tickers()
  if config.environment == "local":
    tickers = get_local_tickers(directory)
  return np.sort(tickers)