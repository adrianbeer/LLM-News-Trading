import logging
import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from src.utils.preprocessing.news_preprocessing import (get_company_abbreviation, 
                                           yahoo_get_wrapper, 
                                           filter_body, 
                                           infer_author,
                                           remove_company_specifics)
from src.config import config
from src.utils.dataframes import block_apply_factory, parallelize_dataframe
from src.utils.time import convert_timezone
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
tqdm.pandas()

NEWS_SOURCES = {
    "FNSPID": config.data.fnspid.raw_html_parsed,
    "BZG": config.data.benzinga.raw_html_parsed
}

def verify_company_name_in_body(df):
    return df.apply(lambda x: bool(re.search(x["short_name"],
                                            x.title + x["body"].replace("( )*\n( )*", " "),
                                            re.IGNORECASE)),
                    axis=1)

def merge_news_sources():
    # Import and merge news bodies from different sources
    dfs = []
    for source in NEWS_SOURCES:
        df = pd.read_parquet(NEWS_SOURCES[source])
        # FNSPID has a intra_day_time boolean column, but BZG doesnt, since BZG always has intra_day timestamps
        if source == "BZG":
            df['intra_day_time'] = True
        dfs.append(df)
        
    ddf = pd.concat(dfs, axis=0)
    print(f"{dfs[0].shape[0]=}(bzg) + {dfs[1].shape[0]=} (fnspid) = {ddf.shape[0]=}")
    ddf = ddf.reset_index(drop=True)

    ddf['time'] = ddf['time'].progress_map(lambda x: convert_timezone(pd.to_datetime(x)))
    
    # Removing all news without intra_day time information, otherwise merging news in the same
    # over night decision segement doesn't work
    is_intra_day_time_news = ~((ddf.time.dt.hour == 0) & (ddf.time.dt.minute == 0) & (ddf.time.dt.second == 0))
    ddf = ddf.loc[is_intra_day_time_news, :].drop(columns=['intra_day_time'])
    
    # Set is_overnight_news to 1... These should not contain as much unprocessed information as real time news
    ddf["is_overnight_news"] = (
        ddf.time.dt.hour >= 16) \
        | (ddf.time.dt.hour <= 9) \
        | ((ddf.time.dt.hour == 9) & ((ddf.time.dt.minute <= 30))
    )

    ## Remove rows for which no stock ticker is recorded
    ddf = ddf[ddf.stocks != '']
    return ddf

def make_ticker_name_mapping():
    all_tickers = None
    for source in NEWS_SOURCES:
        df = pd.read_parquet(NEWS_SOURCES[source], columns=["stocks"])
        if all_tickers is None: 
            all_tickers = df.stocks.drop_duplicates()
        all_tickers = pd.concat([all_tickers, df.stocks.drop_duplicates()], axis=0).drop_duplicates()

    # Tickers sometimes have a dollar sign in front of them.
    # This is common practice to indicate that the acronym refers to a stock ticker.
    # Remove accidental whitespaces before or after tickers
    all_tickers = all_tickers.apply(lambda x: x.strip("$ "))

    # Some stock tickers are in lowercase.
    # Altough yfinance can handle lowercase tickers we uppercase them
    # in order to avoid inconsistencies and avoid duplicates.
    all_tickers = all_tickers.str.upper()
    
    # Colon indicates foreign exchange
    colon_tickers = [x for x in all_tickers.values if ":" in x]
    print(f"Around {len(colon_tickers)} stock tickers are from foreign exchanges. \n"
        f"The list of foreign exchanges is:")
    set([x.split(":")[0] for x in colon_tickers])
    is_foreign_ticker = all_tickers.apply(lambda x: ":" in x)

    # We remove these foreign exchanges:
    all_tickers = all_tickers[~is_foreign_ticker]

    all_tickers.drop_duplicates(inplace=True)

    ### Full-Name-Discovery
    #! This  should not be executed on a cluster because of missing internet connection...
    company_names = all_tickers.progress_map(lambda x: yahoo_get_wrapper(x))
    all_mapper = pd.concat([all_tickers, company_names], axis=1)
    all_mapper.columns = ["ticker", "company_names"]
    print(f"For {all_mapper[all_mapper.isna().any(axis=1)].shape[0]} tickers, yfinance had no entry, or at least not entry for the `longName`")

    # E.g. AIS, PTNR, BSI, LUFK, BFRM. Can't find these stocks on guidants as well and
    # Some are not headquartered in the US.
    all_mapper = all_mapper.dropna()

    mapper = all_mapper
    mapper.set_index("ticker", inplace=True)
    company_endings = pd.read_table(config.data.shared.corporation_endings).iloc[:, 0]
    
    # Apply `get_company_abbreviation` twice in order to get rid of Enterprise, Ltd.
    # Otherwise , Ltd. remains. If no acronym, name stays as is.
    mapper["short_name"] = mapper.company_names.apply(lambda x: get_company_abbreviation(x, company_endings=company_endings))
    mapper = mapper.applymap(lambda x: x.strip(" "))
    
    mapper.to_parquet(config.data.shared.ticker_name_mapper)
    return mapper


def preprocess_news(ddf):
    ### Duplikate Entfernen
    samples_before = ddf.shape[0]
    ddf = ddf.drop_duplicates()
    samples_after = ddf.shape[0]
    print(f"Removing duplicates: {samples_before=}, {samples_after=}")
    
    # Ticker-Company Resolution
    mapper = pd.read_parquet(config.data.shared.ticker_name_mapper)
    ddf = ddf[ddf.stocks.isin(mapper.index.to_list())]
    ddf["company_name"] = ddf.stocks.progress_map(lambda x: mapper.company_names.loc[x]).astype(str)
    ddf["short_name"] = ddf.stocks.progress_map(lambda x: mapper.short_name.loc[x]).astype(str)
    ddf.loc[:, 'short_name'] = ddf.loc[:, 'short_name'].progress_map(lambda x: x.strip('&,. '))
    print(f"Es verbleiben {ddf.shape[0]} Nachrichten, für die wir den Ticker zu einem Firmennamen aufgelösen konnten.")

    ### Firmennamen-Nachrichtenkörper-Verifikation
    mask = parallelize_dataframe(ddf, verify_company_name_in_body, n_cores=os.cpu_count())
    print(f"Filter for company name appearance in msg body."
          f"Around {len(ddf.stocks.unique())} stocks before filtering and {len(ddf[mask].stocks.unique())} after.")
    # Filter out faulty news, for which the company name doesn't occurr in the news body
    ddf = ddf[mask]
    print(f"{ddf.shape[0]} stocks left after removing the stocks.")

    ## Author-Inferenz 
    ddf["inferred_author"] = None
    ddf["inferred_author"] = ddf.body.progress_apply(infer_author)
    ddf = ddf.drop(columns=["author"]).rename(columns={"inferred_author":"author"})
    ddf["author"] = ddf['author'].astype("object").replace({pd.NA:np.nan})

    ## Make reduced ticker name mapping 
    ticker_name_mapper = pd.read_parquet(config.data.shared.ticker_name_mapper)
    ticker_name_mapper_reduced = ddf[["stocks", "company_name", "short_name"]].drop_duplicates(keep="first")
    
    print(f"From {ticker_name_mapper.shape[0]} to {ticker_name_mapper_reduced.shape[0]} tickers (reduced)")

    ## Parsing News Bodies
    ddf["parsed_body"] = parallelize_dataframe(ddf, block_apply_factory(filter_body, axis=1), n_cores=os.cpu_count())        
    
    return ddf, ticker_name_mapper_reduced

WHITELIST = set("abcdefghijklmnopqrstuvwxyz ")

def stripper(x):
    text = x.lower()
    text = ''.join(filter(WHITELIST.__contains__, text))
    text = ' '.join(text.split(' ')) 
    return text

def make_stripped_news(ddf):
    ddf["parsed_body"] = parallelize_dataframe(ddf["parsed_body"], block_apply_factory(stripper), n_cores=os.cpu_count())
    ddf.to_parquet(config.data.news.stripped) 


def get_indices_where_company_name_is_in_title(ddf):
    proc_news: pd.DataFrame = ddf

    comma_appears_in_short_name = proc_news.apply((lambda x: ',' in x['short_name']), axis=1)
    f"{comma_appears_in_short_name.sum() / proc_news.shape[0]:.0%}"    

    comma_appears_in_short_name = proc_news.apply((lambda x: ',' in x['short_name']), axis=1)
    print(f"{comma_appears_in_short_name.sum()=}")
    proc_news.loc[comma_appears_in_short_name, 'short_name'] = proc_news.loc[comma_appears_in_short_name, 'short_name'].apply(lambda x: x.split(',')[0])

    company_name_appears_in_title = proc_news.apply(
        lambda x: (' '.join(x['short_name'].lower().split(' ')[:-1]) in x['title'].lower()) or (x['stocks'].lower() in x['title'].lower()), axis=1
    )
    f"% of news where the company name appears in the title: {company_name_appears_in_title.sum() / proc_news.shape[0]:.0%}"

    valid_indices = proc_news[company_name_appears_in_title].index
    return valid_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='First invoke with --ticker_name_mapping and afterwards with --process_body.')
    parser.add_argument('--ticker_name_mapping', action='store_true',
                        help='Takes quite a long time, and can`t be parallelized much')
    parser.add_argument('--process_body', action='store_true',
                        help='Can be heavilty parallelized. Invoke this after creating the ticker_name_mapping')
    parser.add_argument('--stripper', action='store_true',
                        help='Make stripped news')
    args = parser.parse_args()
    
    if args.ticker_name_mapping:
        print("Creating ticker_name_mapping...")
        make_ticker_name_mapping()
        print("Done")
        
    if args.process_body:
        print("Processing news bodies...")
        ddf = merge_news_sources()
        ddf, ticker_name_mapper_reduced = preprocess_news(ddf)
        ticker_name_mapper_reduced.to_parquet(config.data.shared.ticker_name_mapper_reduced)
        
        # Removing news, where the company name doesnt appear in the title
        idcs = get_indices_where_company_name_is_in_title(ddf)
        ddf = ddf.loc[idcs, :]
        
        # After making sure that the company name appears in the title (relevance) we remove company identifers
        # from the title.
        ddf.loc[:, "title"] = ddf.progress_apply(lambda x: remove_company_specifics(x.title, 
                                                                             x.company_name, 
                                                                             x.short_name, 
                                                                             x.stocks),
                                                 axis=1)
        
        ddf.to_parquet(config.data.news.cleaned)
        
    if args.stripper:
        ddf = pd.read_parquet(config.data.news.cleaned)
        make_stripped_news(ddf)

    print("End.")

