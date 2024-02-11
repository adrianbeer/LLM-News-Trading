import pandas as pd
ticker = "AA"
prices = pd.read_parquet(path=f"D:/data/iqfeed/daily/cleaned/{ticker}_daily.parquet")
prices.index = prices.index.tz_localize("US/Eastern")
prices.head()