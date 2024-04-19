import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
import pytz

eastern = pytz.timezone('US/Eastern')
import concurrent.futures
import os
import sys
from functools import partial

from src.config import config
from src.preprocessing.data_merger_util import (
    get_appropriate_closing_time,
    get_appropriate_entry_time,
    get_primary_ticker,
    merge_ticker_news_with_prices,
)
from src.utils.dataframes import block_apply_factory, parallelize_dataframe


def consolidate_tickers(tickers: pd.Series, ticker_mapper):
    func_ = partial(get_primary_ticker, mapper=ticker_mapper)
    # Overwrite tickers with consolidated ticker, i.e. the ticker of the time series we use to construct input-output pairs
    return parallelize_dataframe(tickers, 
                                 block_apply_factory(func_), 
                                 os.cpu_count())


def import_and_preprocess_news(input_path):
    news = pd.read_parquet(path=input_path, columns=["time", "stocks", "parsed_body", "staleness"])

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


    ticker_mapper_consolidated = pd.read_parquet(config.data.shared.ticker_name_mapper_consolidated)
    news["stocks"] = consolidate_tickers(news["stocks"], ticker_mapper_consolidated)
    news.dropna(inplace=True)  # Some tickers don't exist, they will be converted to NaNs

    return news


def merge_news_with_price_ts(prices_path,
                             news: pd.DataFrame):

    spy: pd.DataFrame = pd.read_parquet(path=f"{prices_path}/SPY_1min.parquet")
    spy.columns = [x.strip("adj_") for x in spy.columns]
    spy.columns = [f"SPY_{x}" for x in spy.columns]

    # keep_columns_from_news = ['stocks']
    # keep_columns = ["est_entry_time",
    #                 "est_exit_time",
    #                 "entry_time",
    #                 "exit_time",
    #                 "r",
    #                 "r_spy", 
    #                 "unadj_entry_open", 
    #                 "entry_is_too_far_apart", 
    #                 "exit_is_too_far_apart",
    #                 "parsed_body"] + keep_columns_from_news


    func = partial(merge_ticker_news_with_prices, spy=spy)

    with tqdm(total=len(news.stocks.unique()), desc="merge_stock", leave=True, position=0) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(func, ticker_news): ticker_news for name, ticker_news in (news.reset_index()
                                                                                                    .groupby("stocks", as_index=False))}
            dfs = []
            for future in concurrent.futures.as_completed(futures):
                merged = future.result()
                dfs.append(merged)
                pbar.update(1)
    news = pd.concat(dfs)

    is_too_far_apart = (news["entry_is_too_far_apart"] | news["exit_is_too_far_apart"])
    news = news[~is_too_far_apart]

    print(f"Filtered rows: {is_too_far_apart.sum()}")
    print(f"{news.shape[0]} news before, {news.dropna().shape[0]} news after dropping NaNs.\n"
        f"NaNs should occurr, when we don't have a price time series when news occurred.")

    news.dropna(inplace=True)
    news.to_parquet(config.data.merged)


def merge_with_daily_indicators(daily_ts_dir_path, merged_path):
    dataset = pd.read_parquet(path=merged_path)
    
    indicators = ["std_252", "dollar_volume", 'r_intra_(t-1)', 'unadj_open', 'cond_vola']
    dataset[indicators] = np.NaN

    tickers = dataset.stocks.unique()
    for ticker in tqdm(tickers):
        prices = pd.read_parquet(path=f"{daily_ts_dir_path}/{ticker}_daily.parquet")
        if prices.empty:
            continue
        
        prices.index = prices.index.tz_localize("US/Eastern")
        
        ticker_dat = (dataset.loc[dataset.stocks == ticker, :]
                            .reset_index()
                            .drop(columns=indicators)
                            .sort_values("est_entry_time"))
        merged = pd.merge_asof(ticker_dat, 
                            prices[indicators].reset_index().rename(columns={"date": "daily_indic_date"}), 
                            left_on="est_entry_time", 
                            right_on="daily_indic_date", 
                            direction="backward")
        
        # If the most recent indicators refer to a time that is too old,
        # we don't use the indicators and leave them as NaNs
        time_discrepancy_filter = (merged.est_entry_time - merged["daily_indic_date"]) >= pd.Timedelta(days=2)
        merged.loc[time_discrepancy_filter, indicators] = np.nan
        
        merged.set_index("index", inplace=True)
        dataset.loc[merged.index, indicators] = merged[indicators]
    dataset = dataset.sort_index()
    dataset.to_parquet(path=merged_path)


if __name__ == "__main__":
    print(f"Starting data_merger: {sys.argv}")
    cmd = sys.argv[1]
    
    if cmd == "initial_merge":
        news = import_and_preprocess_news(input_path=config.data.news.cleaned)
        merge_news_with_price_ts(prices_path=config.data.iqfeed.minute.cleaned,
                                 news=news)

    elif cmd == "merge_daily_indicators":
        merge_with_daily_indicators(daily_ts_dir_path=config.data.iqfeed.daily.cleaned,
                                    merged_path=config.data.merged)
    else:
        raise ValueError(f"Invalid input argument: {cmd}")
    print("Finished data_merger...") 


# ------------ Inspecting staleness
# dataset = pd.read_parquet(config.data.news.cleaned, columns=["time", "stocks", "staleness"])
# print((dataset["staleness"] >= 1).sum())
# print((dataset["staleness"] >= 0.995).sum())
# import plotly.express as px
# px.histogram(dataset["staleness"])
