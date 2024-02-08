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
from tqdm import tqdm

from src.config import config
from src.utils.tickers import get_tickers

tickers = get_tickers(config.data.iqfeed.daily.cleaned)


def get_insample_conditional_volatilies(ts: pd.Series):
    model = (arch_model(ts * 100, 
                        mean = 'Constant', 
                        vol = 'GARCH', 
                        p = 1, o = 0, q = 1)
                .fit(update_freq = 0, disp = 'off'))

    # coefs = model.params
    cond_vol = model.conditional_volatility
    # Downscale after having upscaled for numerical stability 
    cond_vol /= 100
    return cond_vol


def add_indicators(ticker):
    daily_price_path = f"{config.data.iqfeed.daily.cleaned}/{ticker}_daily.parquet"
    prices = pd.read_parquet(path=daily_price_path)
    
    # TODO: train_test split... prevent forward looking bias!
    # Requires train test split before using this module...  
    prices["std_252"] = prices["adj_close"].pct_change(fill_method=None).rolling(252, min_periods=252).std()*(252**0.5)
    prices["r"] = prices["adj_close"] / prices["adj_close"].shift(-1) - 1 

    mask = ~prices["r"].isna()
    prices["cond_vola"] = np.nan
    prices.loc[mask, "cond_vola"] = get_insample_conditional_volatilies(prices[mask, "r"])
    
    prices["dollar_volume"] = prices["adj_volume"] * (prices["adj_close"] + prices["adj_open"])/2
    prices["r_intra_(t-1)"] = (prices["adj_close"] / prices["adj_open"] - 1).shift(periods=1)
    prices["unadj_open"] = prices["adj_open"] / prices["cum_split_ratio"]
    
    prices.to_parquet(path=daily_price_path)
    return prices


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default=False)
    args = parser.parse_args()
    if args.ticker:
        prices = add_indicators(args.ticker)
        print(prices.head())
        
    with tqdm(total=len(tickers), desc="tickers (indicator_applicator)", leave=True, position=0) as pbar:
        # let's give it some more threads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(add_indicators, ticker): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                pbar.update(1)
