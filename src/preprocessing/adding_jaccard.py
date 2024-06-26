from tqdm.auto import tqdm
import pandas as pd 
import numpy as np
from src.config import config
from src.utils.strings import jaccard_similarity


dataset: pd.DataFrame = pd.read_parquet(path=config.data.merged)

# To determine the freshness of news, I compare the similarity of each news article with all articles published in the previous three days.
original_index_name = dataset.index.name
for ticker in tqdm(list(set(dataset.stocks)), desc="stocks"):
    orig_sort_ticker_news = dataset[dataset.stocks == ticker]

    # Using time sorted df makes for easier splicing later
    ticker_news = orig_sort_ticker_news.reset_index().set_index("est_entry_time").sort_index(ascending=True)
    
    # Set staleness of first news message to 0 
    ticker_news.at[ticker_news.index[0], "jaccard"] = 0

    for idx in orig_sort_ticker_news.index:
        time = orig_sort_ticker_news.at[idx, 'est_entry_time']
        previous_news = ticker_news.loc[(time-pd.DateOffset(days=3)):time-pd.DateOffset(minutes=1), "parsed_body"]
        if len(previous_news) == 0:
            ticker_news.at[time, "jaccard"] = 0
        else:
            current_str = orig_sort_ticker_news.at[idx, 'parsed_body']
            # try:
            jaccards = previous_news.apply(lambda x: jaccard_similarity(current_str, x))
            # except ZeroDivisionError as e:
            #     print(e)
            #     print(f"{current_str=}")
            #     print(f"{previous_news=}")
            ticker_news.loc[ticker_news[original_index_name] == idx, "jaccard"] = jaccards.max()
            
    ticker_news.set_index(original_index_name, inplace=True)
    # Add entries to data set
    dataset.loc[ticker_news.index, "jaccard"] = ticker_news.loc[:, "jaccard"]
    
dataset.to_parquet(path=config.data.merged)