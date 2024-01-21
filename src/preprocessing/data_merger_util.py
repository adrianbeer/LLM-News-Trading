
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from dotmap import DotMap
import yaml
import logging

config = DotMap(yaml.safe_load(open("src/config.yaml")), _dynamic=False)
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