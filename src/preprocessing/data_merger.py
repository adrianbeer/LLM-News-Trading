import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px
from src.utils.strings import jaccard_similarity

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



def add_additional_indicators():
    print("Adding additional indicators...")
    dat: pd.DataFrame = pd.read_parquet(path=config.data.merged)

    # Add overnight news tag
    dat["has_intraday_time"] = ~((dat.news_time.dt.hour == 0) & (dat.news_time.dt.minute == 0) & (dat.news_time.dt.seconds == 0))
    # Set is_overnight_news to 1... These should not contain as much unprocessed information as real time news
    dat["is_overnight_news"] = (
        dat.news_time.dt.hour >= 16) \
        | (dat.news_time.dt.hour <= 9) \
        | ((dat.news_time.dt.hour == 9) & ((dat.news_time.dt.minute <= 30))
    )
    
    dat.to_parquet(config.data.merged)


def _same_night_news_jaccard_filter(ddf):
    drop_idcs = []
    for i in list(range(ddf.shape[0]))[:-1]:
        for j in list(range(ddf.shape[0]))[i+1:]:
            jaccard_body = jaccard_similarity(ddf.iloc[i]['parsed_body'], ddf.iloc[j]['parsed_body'])
            if jaccard_body > 0.9:
                drop_idcs.append(ddf.iloc[i].index)
    ddf = ddf.drop(drop_idcs)
    print(f"Dropping {len(drop_idcs)=} news via same_night_news_jaccard_filter function.")
    return ddf


def merge_same_night_news(df: pd.DataFrame) -> pd.Series:
    assert 'title' in df.columns
    if df.shape[0] == 1:
        # Nothing to merge
        return df.iloc[0]
    else:
        # Take the row/ time of the latest news in that overnight segment as the new row template
        #! If we have importance/ relevance tags on the news we might want to prioritice e.g. ad-hocs here 
        df.sort_values('news_time', ascending=False).iloc[0]
        assert df.iloc[0]['news_time'] >= df.iloc[1]['news_time']
        df = _same_night_news_jaccard_filter(df)
        
        merged_row = df.iloc[0]
        merged_row['title'] = ' '.join(df['title'].tolist())
        merged_row['parsed_body'] = ' '.join(df['parsed_body'].tolist())
        #! merge title and parsed body here...
        merged_row['parsed_body'] = ' '.join([merged_row['title'], merged_row['parsed_body']])
        
        return merged_row

def merge_overnight_news_for_stock(df):
    df = df.groupby('est_entry_time').apply(merge_same_night_news)
    return df

def merge_all_overnight_news():
    dat: pd.DataFrame = pd.read_parquet(path=config.data.merged)
    print(f"Before overnight merging {dat.shape[0]=}")
    
    tmp = dat.loc[:, ['est_entry_time', 'r', 'stocks']].groupby(['stocks', 'est_entry_time']).count()
    sum_of_news_sharing_a_segment = tmp.loc[tmp['r'] > 2, 'r'].sum()
    count_of_segments_with_more_than_one_news = tmp[tmp['r'] > 2].shape[0]
    print(f"{sum_of_news_sharing_a_segment / dat.shape[0]:.0%} of news are in the same decision segment!!!")
    print(
        f"{sum_of_news_sharing_a_segment} same segment news will be compressed to {count_of_segments_with_more_than_one_news} segments.." \
        f"\nOn average {sum_of_news_sharing_a_segment/count_of_segments_with_more_than_one_news:.1f} of those news will be compressed to one segment")

    dat = dat.groupby('stocks').apply(merge_overnight_news_for_stock)
    print(f"After overnight merging {dat.shape[0]=}")
    return dat

if __name__ == "__main__":
    news_msg_source = config.data.news.cleaned
    print(f"Starting data_merger: {sys.argv} using {news_msg_source=}")
    cmd = sys.argv[1]
    
    if cmd == "initial_merge":
        news = import_and_preprocess_news(input_path=news_msg_source)
        merge_news_with_price_ts(prices_path=config.data.iqfeed.minute.cleaned,
                                 news=news)
        add_additional_indicators()

    if cmd == 'merge_overnight_news':
        df = merge_all_overnight_news()
        #! This is only temporary for debugging
        df.to_parquet(path="data/debugging_merged_overnight.parquet")
        
    elif cmd == "merge_daily_indicators":
        merge_with_daily_indicators(daily_ts_dir_path=config.data.iqfeed.daily.cleaned,
                                    merged_path=config.data.merged)
    else:
        raise ValueError(f"Invalid input argument: {cmd}")
    print("Finished data_merger...") 
