# Applying Indicators to Price Data

# Takes the output from `iq_feed_cleaning.ipynb` and for each row, adds indicators. \
# This is preferred over calculating the indicators for each timestamp/date, as it saves 
# a lot of computations. \
# It does require us to do a look-up, but so does the alternative.
import concurrent.futures
import os
import argparse

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm

from src.config import config
from src.utils.tickers import get_tickers


def get_insample_conditional_volatilies(ts: pd.Series):
    model = (arch_model(100 * ts, 
                        mean = 'Constant', 
                        vol = 'GARCH', 
                        p = 1, o = 0, q = 1)
                .fit(update_freq = 0, disp = 'off'))

    # coefs = model.params
    cond_vol = model.conditional_volatility
    # Downscale after having upscaled for numerical stability 
    cond_vol /= 100
    return cond_vol


def write_indicators(ticker):
    daily_price_path = f"{config.data.iqfeed.daily.cleaned}/{ticker}_daily.parquet"
    prices = pd.read_parquet(path=daily_price_path)
    prices = add_indicators(prices)
    prices.to_parquet(path=daily_price_path)


indicators = ["std_252", "dollar_volume", 'r_intra_(t-1)', 'cond_vola']

def add_indicators(prices):
    
    if prices.shape[0] == 1:
        # Return empty DataFFrame
        prices.loc[:, indicators] = np.nan
        return prices
    
    # If there are large holes in the time series we apply add_indicators recursively to each connected block 
    timedeltas = prices.index.diff()
    timedelta_mask = timedeltas >= pd.Timedelta("14 days")

    # If unadjosted close is close to 0 we also split the time series, i.e. if there is a period where
    # the time series has appropriate values before and after a certain "penny stock" period
    prices["unadj_open"] = prices["adj_open"] / prices["cum_split_ratio"]
    zero_price_mask = prices["unadj_open"] < 0.5
    
    mask = timedelta_mask | zero_price_mask
    
    if mask.any():
        groupers = mask.cumsum()
        prices = prices.groupby(groupers, group_keys=False).apply(add_indicators)
        return prices

    # TODO: train_test split... prevent forward looking bias!
    # Requires train test split before using this module...  
    prices["std_252"] = prices["adj_close"].pct_change(fill_method=None).rolling(252, min_periods=252).std()*(252**0.5)
    prices["r"] = prices["adj_close"] / prices["adj_close"].shift(-1) - 1 

    mask = ~prices["r"].isna()
    prices["cond_vola"] = np.nan
    prices.loc[mask, "cond_vola"] = get_insample_conditional_volatilies(prices.loc[mask, "r"])
    
    prices["dollar_volume"] = prices["adj_volume"] * (prices["adj_close"] + prices["adj_open"])/2
    prices["r_intra_(t-1)"] = (prices["adj_close"] / prices["adj_open"] - 1).shift(periods=1)
    
    return prices


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default=False)
    args = parser.parse_args()
    if args.ticker:
        prices = write_indicators(args.ticker)
        print(prices.head())
        exit()

    tickers = get_tickers(config.data.iqfeed.daily.cleaned)
    with tqdm(total=len(tickers), desc="tickers (indicator_applicator)", leave=True, position=0) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(write_indicators, ticker): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(futures):
                # This line will propagate any exceptions
                check_for_exceptions = future.result()
                
                ticker = futures[future]
                pbar.update(1)
