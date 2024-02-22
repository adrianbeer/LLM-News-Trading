import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import config
from src.preprocessing.news_parser import filter_body, infer_author
from src.utils.dataframes import block_apply_factory, parallelize_dataframe
from src.utils.time import convert_timezone

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
tqdm.pandas()



def verify_company_name_in_body(df):
    return df.apply(lambda x: bool(re.search(x["short_name"],
                                            x.title + x["body"].replace("( )*\n( )*", " "),
                                            re.IGNORECASE)),
                    axis=1)


if __name__ == "__main__":
    ## Import and merge news bodies from different sources
    fnspid_news = pd.read_parquet(config.data.fnspid.raw_html_parsed)
    bzg_news = pd.read_parquet(config.data.benzinga.raw_html_parsed)

    bzg_news['intra_day_time'] = True
    ddf = pd.concat([fnspid_news, bzg_news], axis=0)
    print(f"{bzg_news.shape[0]}(bzg) + {fnspid_news.shape[0]} (fnspid) = {ddf.shape[0]}")
    ddf = ddf.reset_index(drop=True)

    ## Author-Inferenz 

    # Remove rows for which noo stock ticker is recorded
    ddf = ddf[ddf.stocks != '']

    ddf["inferred_author"] = None
    ddf["inferred_author"] = ddf.body.progress_apply(infer_author)

    ddf = ddf.drop(columns=["author"]).rename(columns={"inferred_author":"author"})
    ddf["author"] = ddf['author'].astype("object").replace({pd.NA:np.nan})

    # Ticker-Company Resolution
    mapper = pd.read_parquet(config.data.shared.ticker_name_mapper)
    ddf = ddf[ddf.stocks.isin(mapper.index.to_list())]
    ddf["company_name"] = ddf.stocks.progress_map(lambda x: mapper.company_names.loc[x]).astype(str)
    ddf["short_name"] = ddf.stocks.progress_map(lambda x: mapper.short_name.loc[x]).astype(str)
    print(f"Es verbleiben {ddf.shape[0]} Nachrichten, für die wir den Ticker zu einem Firmennamen aufgelösen konnten.")

    ### Duplikate Entfernen
    samples_before = ddf.shape[0]
    ddf = ddf.drop_duplicates()
    samples_after = ddf.shape[0]
    print(f"{samples_before=}, {samples_after=}")

    ### Firmennamen-Nachrichtenkörper-Verifikation
    mask = parallelize_dataframe(ddf, verify_company_name_in_body, n_cores=os.cpu_count())

    print(f"Around {len(ddf.stocks.unique())} stocks before filtering and {len(ddf[mask].stocks.unique())} after")

    # Filter out faulty news, for which the company name doesn't occurr in the news body
    ddf = ddf[mask]
    print(f"{ddf.shape[0]} stocks left.")

    ## Make reduced ticker name mapping 
    ticker_name_mapper = pd.read_parquet(config.data.shared.ticker_name_mapper)
    ticker_name_mapper_reduced = ddf[["stocks", "company_name", "short_name"]].drop_duplicates(keep="first")
    ticker_name_mapper_reduced.to_parquet(config.data.shared.ticker_name_mapper_reduced)
    print(f"From {ticker_name_mapper.shape[0]} to {ticker_name_mapper_reduced.shape[0]} tickers (reduced)")


    ## Parsing News Bodies
    ddf["time"] = ddf["time"].progress_map(lambda x: convert_timezone(pd.to_datetime(x)))
    sample = ddf
    ddf["parsed_body"] = parallelize_dataframe(sample, block_apply_factory(filter_body, axis=1), n_cores=os.cpu_count())
    ddf.to_parquet(config.data.news.cleaned)

