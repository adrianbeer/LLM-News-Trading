from benzinga import news_data, financial_data
import pandas as pd 
import json
import html2text
from bs4 import BeautifulSoup
import pickle
import time

api_key = "63bcced6972b4b88b66b0e669336555f"
fin = financial_data.Benzinga(api_key)
paper = news_data.News(api_key)

with open("data/tickers.pkl",'rb') as f:
    TICKERS = pickle.load(f)

# "News"
RELEVANT_CHANNELS = ["Earnings", "Dividends", "Financing", "Events"]


def parse_story_to_row(story):
    # Converts JSON (dict) story to a list 
    tags = [d["name"] for d in story["tags"]]
    channels = [d["name"] for d in story["channels"]]

    if len(set(channels).intersection(set(RELEVANT_CHANNELS))) == 0: 
        return None
    
    stocks = [d["name"] for d in story["stocks"]]
    if len(stocks) != 1: 
        return None
    
    stocks = stocks[0] # This is the ticker(s)
    if stocks not in TICKERS.categories: return None # Don't process, if we have no stock data for it

    body = story["body"]
    time = pd.to_datetime(story["created"])
    title = story["title"]
    author = story["author"]
    id = story["id"]
    return id, title, tags, json.dumps(channels), stocks, body, body_formatter(body), time, author


def body_formatter(body):
    soup = BeautifulSoup(body, features="html.parser")
    for t in soup.find_all('table'):
        t.decompose()

    new_body = str(soup)
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    # h.bypass_tables = True
    # h.ignore_emphasis = True
    h.drop_white_space = True
    return h.handle(new_body)


if __name__ == "__main__":
    # Initialize empty data frame
    story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "body", "html_body"], dtype=object)
    story_df = story_df.astype({"stocks": TICKERS})
    
    no_error = True
    # `i`  together with `pagesize` specifies how many stories entries will be downloaded
    # Not all these stories will be valid news, so the size of the resulting data frame has to be checked
    i = 0 
    pagesize = 100
    while no_error:
        try:
            time.sleep(2)
            print(i)
            #  channel="Earnings"
            stories = paper.news(display_output="full", page=i, channel=",".join(RELEVANT_CHANNELS), pagesize=pagesize)
            for s in range(len(stories)):
                story = stories[s]
                parsed_story = parse_story_to_row(story)
                if parsed_story is None: continue
                id, title, tags, channels, stocks, html_body, body, timestamp, author = parsed_story
                #if "Earnings" in channels:
                story_df.loc[id, ["time", "stocks", "author", "title", "channels", "body", "html_body"]] = timestamp, stocks, author, title, channels, body, html_body
            i += 1
        except Exception as e:
            print(e) 
            no_error = False
    print(f"#rows: {story_df.shape[0]}")

    # story_df = story_df.astype({"author": "category", "channels": "category"})
    print(story_df.dtypes)

    with open("data/story_df_raw.pkl", "wb") as f:
        pickle.dump(story_df, f)

