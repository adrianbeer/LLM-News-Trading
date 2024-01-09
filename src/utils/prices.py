
import yfinance 
import numpy as np
import pandas as pd

def calc_backward_adjustment_factors(ticker: str, return_dataframe: bool = False):
    """Calculates the backward adjust factors based on data from yfinance.

    `cum_split_ratio` and `backward_adjustment_factor` are synonymous.
    IMPORTANT !!!!!!!!!!
    In yahoo finance the `Close` is adj. for splits and the `Adj. Close` for splits and dividends
    Dividends on Yahoo Finance are adjusted for splits!
    """
    df = yfinance.download(ticker, period="max",actions=True)
    if ticker in yfinance.shared._ERRORS:
      return None
    df.sort_index(ascending=True, inplace=True)
    df["split_ratio"] = 1

    # Add splits
    df["Stock Splits"] = df["Stock Splits"].shift(-1, fill_value=1)
    split_mask = df["Stock Splits"] > 0
    df.loc[split_mask, "split_ratio"] = 1 / df["Stock Splits"]

    # Add dividends
    dividend_mask = (df["Dividends"] > 0)
    # The Close is split, but not dividend adjusted
    df.loc[dividend_mask, "split_ratio"] = (1 - df.loc[dividend_mask, "Dividends"].values / df.shift(1).loc[dividend_mask, "Close"].values)

    df["cum_split_ratio"] = np.cumprod(df["split_ratio"][::-1])[::-1]
    if return_dataframe:
      return df
    else:
      return df["cum_split_ratio"]
    
    
def calc_adj_prices(prices: pd.DataFrame, bafs: pd.Series):
  X = bafs.index.tz_localize("US/Eastern")
  # danger: bafs was passed as reference, not valuee.
  bafs.index = X  + pd.DateOffset(hours=16, minutes=1)
  
  prices_adj = pd.merge_asof(prices, bafs, left_index=True, right_on="Date", direction="forward")
  prices_adj.loc[:, ["open", "high", "low", "close"]] = prices_adj[["open", "high", "low", "close"]].mul(prices_adj.cum_split_ratio, axis="index")
  prices_adj.loc[:, "volume"] = prices_adj["volume"].div(prices_adj["cum_split_ratio"], axis="index")
  prices_adj.drop(columns="Date", inplace=True)
  
  prices_adj.rename(columns=dict(
      zip(["open", "high", "low", "close", "volume"],
          [f"adj_{x}" for x in ["open", "high", "low", "close", "volume"]])
      ),
                    inplace=True)
  return prices_adj