from benzinga import news_data, financial_data
import pandas as pd 
import json
import html2text
from bs4 import BeautifulSoup
import pickle
import time

api_key = "272764ec2a3b4c3188dbef1310a699e7"
fin = financial_data.Benzinga(api_key)
paper = news_data.News(api_key)

with open("data/tickers.pkl",'rb') as f:
    TICKERS = pickle.load(f)

# "News"
RELEVANT_CHANNELS = ["Earnings", "Dividends", "Financing", "Events"]

NO_CHANNEL_TAG_CUTOFF_DATE = pd.Timestamp(year=2016, month=12, day=31)

def parse_story_to_row(story):
    # Converts JSON (dict) story to a list 
    tags: list  = [d["name"] for d in story["tags"]]
    channels: list = [d["name"] for d in story["channels"]]

    time = pd.to_datetime(story["created"])
    stocks = [d["name"] for d in story["stocks"]]

    if time.date() > NO_CHANNEL_TAG_CUTOFF_DATE.date() and (len(set(channels).intersection(set(RELEVANT_CHANNELS))) == 0): 
        return None
    if len(stocks) != 1: 
        return None
    
    ticker = stocks[0] # This is the ticker(s)
    if ticker not in TICKERS.categories: return None # Don't process, if we have no stock data for it

    body = story["body"]
    title = story["title"]
    author = story["author"]
    id = story["id"]

    return id, title, tags, json.dumps(channels), ticker, body, body_formatter(body), time, author


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


def save_df_and_return_empty(story_df, current_year):
    print(f"#rows: {story_df.shape[0]}")

    with open(f"data/story_df_raw_{current_year}.pkl", "wb") as f:
        pickle.dump(story_df, f)

    story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "body", "html_body"], dtype=object)
    story_df = story_df.astype({"stocks": TICKERS})   
    return story_df


if __name__ == "__main__":
    # Initialize empty data frame
    story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "body", "html_body"], dtype=object)
    story_df = story_df.astype({"stocks": TICKERS})
    
    
    # `i`  together with `pagesize` specifies how many stories entries will be downloaded
    # Not all these stories will be valid news, so the size of the resulting data frame has to be checked
    
    
    # 100 is the max pagesize.
    pagesize = 100

    current_year = 0

    # Before 2017 there are no channels!
    ranger_time = pd.date_range(start="2015-01-01", end="2016-12-31", freq="W") 
    ranger = [x.strftime("%Y-%m-%d") for x in ranger_time]

    for r in range(len(ranger) + 1):
        i = 0
        no_error = 0
        while no_error <= 2:
            try:
                time.sleep(2)
                print(i)
                if ranger_time[r+1] <= NO_CHANNEL_TAG_CUTOFF_DATE:
                    stories = paper.news(display_output="full", page=i, date_from=ranger[r], date_to=ranger[r+1], pagesize=100)
                else:
                    stories = paper.news(display_output="full", page=i, date_from=ranger[r], date_to=ranger[r+1], pagesize=100, channel=",".join(RELEVANT_CHANNELS))
                print(f"Number of stories {len(stories)}")
                if len(stories) == 0: 
                    no_error += 1
                for s in range(len(stories)):
                    story = stories[s]
                    parsed_story = parse_story_to_row(story)

                    if parsed_story is None: continue
                    id, title, tags, channels, stocks, html_body, body, timestamp, author = parsed_story
                    
                    year = timestamp.year

                    if current_year != year:
                        story_df = save_df_and_return_empty(story_df, current_year)
                        current_year = year
                        print(year)

                    story_df.loc[id, ["time", "stocks", "author", "title", "channels", "body", "html_body"]] = timestamp, stocks, author, title, channels, body, html_body

                i += 1
            except Exception as e:
                print(e) 
                no_error += 1

    print(f"#rows: {story_df.shape[0]}")
    with open(f"data/story_df_raw_{current_year}.pkl", "wb") as f:
        pickle.dump(story_df, f)

    print(story_df.dtypes)



