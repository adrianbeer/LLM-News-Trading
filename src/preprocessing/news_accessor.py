from benzinga import news_data, financial_data
import pandas as pd 
import json
import html2text
from bs4 import BeautifulSoup

def parse_story_to_row(story):
    tags = [d["name"] for d in story["tags"]]
    channels = [d["name"] for d in story["channels"]]
    stocks = [d["name"] for d in story["stocks"]]
    assert len(stocks) == 1
    stocks = stocks[0]
    body = story["body"]
    time = story["created"]
    title = story["title"]
    author = story["author"]
    id = story["id"]
    return id, title, tags, channels, stocks, body, time, author

api_key = "63bcced6972b4b88b66b0e669336555f"
fin = financial_data.Benzinga(api_key)
paper = news_data.News(api_key)

def body_formatter(body):
    
    soup = BeautifulSoup(body)
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

stories = paper.news(display_output="full", page=1, pagesize=5, channel="Earnings")
story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "body", "html_body"], dtype=object)
for i in range(len(stories)):
    story = stories[i]
    id, title, tags, channels, stocks, body, time, author = parse_story_to_row(story)
    #if "Earnings" in channels:
    story_df.loc[id, "time"] = time
    story_df.loc[id, "stocks"] = stocks
    story_df.loc[id, "author"] = author
    story_df.loc[id, "title"] = title
    story_df.loc[id, "channels"] = channels
    story_df.loc[id, "body"] = body_formatter(body)
    story_df.loc[id, "html_body"] = body

print(story_df.shape)
print(story_df.head(10))
story_df.to_csv("data/stories.csv")
#print(story_df[story_df.channels.apply(lambda x: "Earnings" in x)]["body"].iloc[0])