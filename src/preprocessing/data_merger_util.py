
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from dotmap import DotMap
import yaml
import logging
from src.config import config

nyse_cal = mcal.get_calendar('NYSE')

def get_next_available_candle(prices: pd.DataFrame, 
                              time: pd.Timestamp) -> pd.Series:
    entry_candle_idx = prices.index.get_indexer(target=[time], 
                                                method="bfill")
    entry_candle = prices.take(entry_candle_idx).iloc[0]
    return entry_candle


def get_next_valid_day_for_time(time, valid_days):
    i = 1
    while True:
        new_time = time + pd.DateOffset(days=i)
        if new_time.date() in valid_days:
            return new_time
        if i == 7:
            return ValueError()
        i += 1


def get_appropriate_closing_time(time: pd.Timestamp, tz="US/Eastern") -> pd.Timestamp:
    close_time = pd.Timestamp(year=time.year, month=time.month, day=time.day, hour=16, minute=1, tz=tz)
    valid_days = [x.date() for x in nyse_cal.valid_days(start_date=time.date(), end_date=time.date() + pd.DateOffset(days=10))]

    if (time.date() in valid_days) and ((time.hour < 9) or ((time.hour == 9) and (time.minute < 30))):
        return close_time
    
    return get_next_valid_day_for_time(close_time, valid_days)


def get_appropriate_entry_time(time: pd.Timestamp, tz="US/Eastern") -> pd.Timestamp:
    open_time = pd.Timestamp(year=time.year, month=time.month, day=time.day, hour=9, minute=31, tz=tz)
    valid_days = [x.date() for x in nyse_cal.valid_days(start_date=time.date(), end_date=time.date() + pd.DateOffset(days=10))]
    
    if (time.date() in valid_days) and ((time.hour < 9) or ((time.hour == 9) and (time.minute < 30))):
        return open_time
    elif (time.hour >= 16) or (time.date() not in valid_days):
        return get_next_valid_day_for_time(open_time, valid_days)
    else:
        return time.ceil("min")  + pd.Timedelta(minutes=1)
    

def get_primary_ticker(ticker, mapper):
    company_name = mapper.loc[mapper["stocks"] == ticker, "company_name"]
    if len(company_name) == 0:
        # No matching entry
        return None
    else:
        company_name = company_name.iat[0]
    primary_ticker = mapper.loc[(mapper["company_name"] == company_name) & mapper.is_primary_ticker, "stocks"].iat[0]
    return primary_ticker


def merge_ticker_news_with_prices(ticker_news: pd.DataFrame, spy: pd.DataFrame):
    ticker = ticker_news.stocks.iat[0]
    prices: pd.DataFrame = pd.read_parquet(path=f"{config.data.iqfeed.minute.cleaned}/{ticker}_1min.parquet")
    prices.columns = [x.strip("adj_") for x in prices.columns]
    prices = prices.reset_index().sort_values("time")

    # We generally neeed to use `merge_asof` here instead of simple `merge`, because
    # Sometimes no auction occurred or was recorded at 16:00 or things of this sort.

    # Left key must be sorted
    ticker_news.sort_values("est_entry_time", inplace=True)
    merged = pd.merge_asof(ticker_news, prices.rename(columns=dict(time="entry_time")), left_on="est_entry_time", right_on="entry_time", direction="forward")

    merged.sort_values("est_exit_time", inplace=True)
    merged = pd.merge_asof(merged, prices.rename(columns=dict(time="exit_time")), left_on="est_exit_time", right_on="exit_time", suffixes=("_entry", "_exit"), direction="backward")
    # We use the O part of the OHLC for intra day candles here for convenienece as well
    merged["r"] = merged["open_exit"] / merged["open_entry"] - 1

    # Ideally we do this for every stock first and then we come back with the complete dataframe... (depends on if it fits in memory)
    # Merge news and stock prices with spy prices
    merged.sort_values("entry_time", inplace=True)
    merged.dropna(inplace=True) # NaN can occurr e.g. if there ist not exit_time for an est_exit_time
    merged = pd.merge_asof(merged, spy, left_on="entry_time", right_on="time", direction="forward")

    # TODO: Don't use intraday as exit here (closing candle) but the actual closing auction...
    # But for that we need the daily time series, not with minute frequency
    merged.sort_values("exit_time", inplace=True)
    merged.dropna(inplace=True)
    merged = pd.merge_asof(merged, spy, left_on="exit_time", right_on="time", suffixes=("_entry", "_exit"), direction="backward")

    # Calculate to potentially filter out penny stocks later on
    merged["unadj_entry_open"] = merged["open_entry"] / merged["cum_split_ratio_entry"]

    #TODO: shouldnt we use open entry and close exit?
    merged["r_spy"] = merged["SPY_close_exit"] / merged["SPY_close_entry"] - 1

    merged.set_index("index", inplace=True)
    
    # Filter out stocks where estimated entry/exit is further apart than actual entry/exit by more than 1h
    merged["entry_is_too_far_apart"] = (merged.entry_time - merged.est_entry_time) > pd.Timedelta(hours=1)
    merged["exit_is_too_far_apart"] = (merged.exit_time - merged.est_exit_time) > pd.Timedelta(hours=1)
    
    return merged

