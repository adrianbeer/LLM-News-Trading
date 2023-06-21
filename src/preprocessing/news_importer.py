from benzinga import news_data, financial_data
import pandas as pd 
import json
import html2text
from bs4 import BeautifulSoup
import pickle

api_key = "63bcced6972b4b88b66b0e669336555f"
fin = financial_data.Benzinga(api_key)
paper = news_data.News(api_key)

all_stories = []

RELEVANT_CHANNELS = ["Earnings", "Dividends", "News", "Financing", "Events"]

# source = open("data/tickers.pkl", 'rb').read()
# tickers = json.loads(source)

if __name__ == "__main__":
    no_error = True
    i = 0
    while no_error and i <= 2:
        try:
            #  channel="Earnings"
            stories = paper.news(display_output="full", page=i, channel=",".join(RELEVANT_CHANNELS), pagesize=1000)
            all_stories.append(stories)
            i += 1
        except Exception as e:
            print(e) 
            no_error = False

    print(f"Pagesize: {len(all_stories[0])}")
    with open("data/stories_raw.pkl", "wb") as f:
        pickle.dump(all_stories, f)
