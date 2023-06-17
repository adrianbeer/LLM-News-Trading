from benzinga import news_data, financial_data
import pandas as pd 
import json
import html2text
from bs4 import BeautifulSoup
import pickle
import itertools

RELEVANT_CHANNELS = ["Earnings", "Dividends", "News", "Financing", "Events"]

def parse_story_to_row(story):
    tags = [d["name"] for d in story["tags"]]
    channels = [d["name"] for d in story["channels"]]
    if len(set(channels).intersection(set(RELEVANT_CHANNELS))) == 0: 
        return None
    stocks = [d["name"] for d in story["stocks"]]
    if len(stocks) != 1: 
        return None
    stocks = stocks[0]
    body = story["body"]
    time = story["created"]
    title = story["title"]
    author = story["author"]
    id = story["id"]
    return id, title, tags, channels, stocks, body, time, author

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

with open("data/stories_raw.pkl", 'rb') as f:
    all_stories = pickle.load(f)




stories = list(itertools.chain(*all_stories))
print(f"Total stories: {len(stories)}")

story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "body", "html_body"], dtype=object)
for i in range(len(stories)):
    story = stories[i]
    parsed_story = parse_story_to_row(story)
    if parsed_story is None: continue
    id, title, tags, channels, stocks, body, time, author = parsed_story
    #if "Earnings" in channels:
    story_df.loc[id, "time"] = time
    story_df.loc[id, "stocks"] = stocks
    story_df.loc[id, "author"] = author
    story_df.loc[id, "title"] = title
    story_df.loc[id, "channels"] = json.dumps(channels)
    story_df.loc[id, "body"] = body_formatter(body)
    story_df.loc[id, "html_body"] = body

print(f"Filtered stories: {story_df.shape[0]}")

print(story_df.shape)
print(story_df.head(10))
story_df.to_csv("data/stories.csv")
#print(story_df[story_df.channels.apply(lambda x: "Earnings" in x)]["body"].iloc[0])