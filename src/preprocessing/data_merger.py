import pandas as pd 
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import dask.dataframe as dd
import pytz
eastern = pytz.timezone('US/Eastern')
from functools import partial

from src.config import config, MODEL_CONFIG
from src.preprocessing.data_merger_util import (get_appropriate_closing_time,
                                                get_appropriate_entry_time, 
                                                get_primary_ticker,
                                                merge_ticker_news_with_prices)
from src.utils.dataframes import parallelize_dataframe, block_apply_factory
import os
import concurrent.futures
from functools import partial
from src.utils.tickers import get_tickers

## ------------------------------Import and Preprocess News 

def import_and_preprocess_news():
    news = pd.read_parquet(path=config.data.benzinga.cleaned, columns=["time", "stocks", "parsed_body", "staleness"])

    # (old comment?) Necessary to get `us` units, otherwise pandas will always convert back to `ns` for some reason.
    news["time"] = news.time.dt.tz_convert(eastern).astype('datetime64[ns, US/Eastern]')
    news.rename(columns={"time":"news_time"}, inplace=True)

    # TODO: This can be *improved* by saying that if we are very close to completing the minute e.g. :55, 
    # then we dont take the next candle (T+1), but the candle after the next(T+2).
    # Watch out, news time is accurate, but candles are right labeled, hence add one minute.
    news["est_entry_time"] = parallelize_dataframe(news["news_time"], 
                                                    block_apply_factory(get_appropriate_entry_time), 
                                                    os.cpu_count())
    news["est_exit_time"] = parallelize_dataframe(news["news_time"], 
                                                block_apply_factory(get_appropriate_closing_time), 
                                                os.cpu_count())

    ##------------------------------ Consolidate Tickers

    ticker_mapper_consolidated = pd.read_parquet("data_shared/ticker_name_mapper_consolidated.parquet")

    # Overwrite tickers with consolidated ticker, i.e. the ticker of the time series we use to construct input-output pairs
    # news["stocks"] = news.stocks.progress_map(lambda ticker: get_primary_ticker(ticker, mapper=ticker_mapper_consolidated))

    func_ = partial(get_primary_ticker, mapper=ticker_mapper_consolidated)
    news["stocks"] = parallelize_dataframe(news["stocks"], 
                                        block_apply_factory(func_), 
                                        os.cpu_count())

    # Some tickers don't exist, they will be converted to NaNs
    news.dropna(inplace=True)

    news.to_pickle("data/temporary_news_df.pkl")


def merge_news_with_price_ts():
    news = pd.read_pickle("data/temporary_news_df.pkl")

    spy: pd.DataFrame = pd.read_parquet(path=f"{config.data.iqfeed.minute.cleaned}/SPY_1min.parquet")
    spy.columns = [x.strip("adj_") for x in spy.columns]
    spy.columns = [f"SPY_{x}" for x in spy.columns]

    keep_columns_from_news = ["staleness"]
    keep_columns = ["est_entry_time",
                    "est_exit_time",
                    "entry_time",
                    "exit_time",
                    "r",
                    "r_spy", 
                    "unadj_entry_open", 
                    "entry_is_too_far_apart", 
                    "exit_is_too_far_apart"] + keep_columns_from_news


    func = partial(merge_ticker_news_with_prices, spy=spy)

    with tqdm(total=len(news.stocks.unique()), desc="merge_stock", leave=True, position=0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(func, ticker_news): ticker_news for name, ticker_news in (news.reset_index()
                                                                                                    .groupby("stocks", as_index=False))}
            dfs = []
            for future in concurrent.futures.as_completed(futures):
                merged = future.result()
                dfs.append(merged[keep_columns])
                pbar.update(1)
    news = pd.concat(dfs)

    mask = (news["entry_is_too_far_apart"] | news["exit_is_too_far_apart"])
    news = news[~(news["entry_is_too_far_apart"]|news["exit_is_too_far_apart"])]

    print(f"Filtered rows: {mask.sum()}")

    print(f"{news.shape[0]} news before. {news.dropna().shape[0]} news after dropping NaNs."
        f"NaNs should occurr, when we don't have a price time series when news occurred.")
    news.dropna(inplace=True)

    # Save to Disk
    news.to_parquet(config.data.merged)


def merge_with_daily_indicators():
    tickers = get_tickers(config.data.iqfeed.daily.cleaned)
    dataset = pd.read_parquet(path=config.data.merged)

    dataset[["std_252", "dollar_volume", 'r_intra_(t-1)', 'unadj_open']] = np.NaN
    indicators = ["std_252", "dollar_volume", 'r_intra_(t-1)', 'unadj_open']

    for ticker in tqdm(tickers):
        prices = pd.read_parquet(path=f"{config.data.iqfeed.daily.cleaned}/{ticker}_daily.parquet")
        prices.index = prices.index.tz_localize("US/Eastern")
        ticker_dat = (dataset.loc[dataset.stocks == ticker, :]
                            .reset_index()
                            .drop(columns=indicators)
                            .sort_values("est_entry_time"))
        merged = pd.merge_asof(ticker_dat, 
                            prices[indicators], 
                            left_on="est_entry_time", 
                            right_on="date", 
                            direction="backward")
        merged.set_index("index", inplace=True)
        dataset.loc[merged.index, indicators] = merged[indicators]
    dataset.to_parquet(path=config.data.merged)

if __name__ == "__main__":
    import_and_preprocess_news()
    merge_news_with_price_ts()
    merge_with_daily_indicators()



# ------------ Inspecting staleness
# dataset = pd.read_parquet(config.data.benzinga.cleaned, columns=["time", "stocks", "staleness"])
# print((dataset["staleness"] >= 1).sum())
# print((dataset["staleness"] >= 0.995).sum())
# import plotly.express as px
# px.histogram(dataset["staleness"])