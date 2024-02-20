import pandas as pd
from src.config import config
from preprocessing.news_parser import get_company_abbreviation, yahoo_get_wrapper
from tqdm import tqdm
tqdm.pandas()


INPUT_DF_PATH = config.data.benzinga.raw_html_parsed
CORPORATION_ENDINGS_FILE = config.data.shared.corporation_endings
OUTPUT_DF_PATH = config.data.shared.ticker_name_mapper

if __name__ == '__main__':
    ddf = pd.read_parquet(INPUT_DF_PATH)
    
    all_tickers = ddf.stocks.unique()

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
    company_names = all_tickers.progress_map(lambda x: yahoo_get_wrapper(x))
    all_mapper = pd.concat([all_tickers, company_names], axis=1)
    all_mapper.columns = ["ticker", "company_names"]
    print(f"For {all_mapper[all_mapper.isna().any(axis=1)].shape[0]} tickers, yfinance had no entry, or at least not entry for the `longName`")

    # E.g. AIS, PTNR, BSI, LUFK, BFRM. Can't find these stocks on guidants as well and
    # Some are not headquartered in the US.
    all_mapper = all_mapper.dropna()

    mapper = all_mapper
    mapper.set_index("ticker", inplace=True)
    company_endings = pd.read_table(CORPORATION_ENDINGS_FILE).iloc[:, 0]
    
    # Apply `get_company_abbreviation` twice in order to get rid of Enterprise, Ltd.
    # Otherwise , Ltd. remains. If no acronym, name stays as is.
    mapper["short_name"] = mapper.company_names.apply(lambda x: get_company_abbreviation(x, company_endings=company_endings))
    mapper = mapper.applymap(lambda x: x.strip(" "))
    
    mapper.to_parquet(OUTPUT_DF_PATH)