"""
Handling of multiple tickers for a the same company.
If there is only one price time series available for the company, we simply group together the tickers.
However in some cases we will have multiple price time series for the same company.

E.g. in case of Alphabet (Google) we have two different tickers and two different stock prices for the same
underlying company. Here `GOOG` and `GOOGL` describe two different classes of stock for the same company.
In this case we will try to only look at the main class. 

We find this class by choosing the Symbol with the longer stock price history, assuming that the history
of it includes(!) the history of the other one completely.
If one time series doesn't include the other we merge the two time series. Ideally based on which time series has more liquidity 
in a given week or but we will simply decide that the newer time series takes precedence for simplicity. 
"""
import pandas as pd
import numpy as np
from src.config import config
from tqdm import tqdm

def get_total_historical_dollar_volume(ticker):
    try:
        price : pd.DataFrame = pd.read_parquet(path=f"{config.data.iqfeed.daily.cleaned}/{ticker}_daily.parquet")
    except FileNotFoundError:
        return np.NaN
    total_historical_dollar_volume = price.dollar_volume.sum()
    return total_historical_dollar_volume
    
    
def get_time_interval(ticker):
    try:
        price : pd.DataFrame = pd.read_parquet(path=f"{config.data.iqfeed.minute.cleaned}/{ticker}_1min.parquet")
    except FileNotFoundError:
        return (np.NaN, np.NaN)
    time_interval = price.index.min(), price.index.max()
    return time_interval


def consolidate_tickers(df: pd.DataFrame) -> pd.DataFrame:
    # We want only one ticker per company name
    # Choose the one that has the most recent values
    df = df.sort_values("total_historical_dollar_volume", descending=True)
    df.loc[df.index[0], "is_primary_ticker"] = True
    return df


if __name__ == "__main__":
    ticker_mapper = pd.read_parquet("data_shared/ticker_name_mapper_reduced.parquet")
    
    ticker_mapper[["first_date", "last_date"]] = np.NaN
    for i in tqdm(ticker_mapper.index):
        ticker = ticker_mapper.loc[i, "stocks"]
        # Adding first_date and last_date is kind of not needed anymore but still interesting... even though big holes can exist... 
        ticker_mapper.loc[i, ["first_date", "last_date"]] = get_time_interval(ticker)
        ticker_mapper.loc[i, "total_historical_dollar_volume"] = get_total_historical_dollar_volume(ticker)
    
    ticker_mapper.dropna(inplace=True)
    ticker_mapper[["first_date", "last_date"]] = ticker_mapper[["first_date", "last_date"]].apply(pd.to_datetime, axis=0)

    ticker_mapper_consolidated = ticker_mapper.copy(deep=True)
    ticker_mapper_consolidated["is_primary_ticker"] = False
    ticker_mapper_consolidated = ticker_mapper_consolidated.groupby("company_name", as_index=False).apply(consolidate_tickers)
    
    print(f"{ticker_mapper_consolidated.shape[0]} entries before consolidation. {ticker_mapper_consolidated[ticker_mapper_consolidated.is_primary_ticker].shape} entries after.")
    
    ticker_mapper_consolidated.to_parquet("data_shared/ticker_name_mapper_consolidated.parquet")