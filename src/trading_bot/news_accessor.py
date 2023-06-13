from benzinga import news_data, financial_data
import pandas as pd 
import json

def parse_story_to_row(story):
    tags = [d["name"] for d in story["tags"]]
    channels = [d["name"] for d in story["channels"]]
    stocks = [d["name"] for d in story["stocks"]]
    body = story["body"]
    time = story["created"]
    title = story["title"]
    author = story["author"]
    id = story["id"]
    return id, title, tags, channels, stocks, body, time, author

api_key = "680260dc809f4e2c93f37fd8234f791a"
fin = financial_data.Benzinga(api_key)
paper = news_data.News(api_key)

stories = paper.news(display_output="full", page=1, pagesize=200, channel="Earnings")
story_df = pd.DataFrame(columns=["time", "author", "title", "channels", "body"], dtype=object)
for i in range(len(stories)):
    story = stories[i]
    id, title, tags, channels, stocks, body, time, author = parse_story_to_row(story)
    story_df.loc[id, "time"] = time
    story_df.loc[id, "author"] = author
    story_df.loc[id, "title"] = title
    story_df.loc[id, "channels"] = channels
    story_df.loc[id, "body"] = body

print(story_df.shape)
print(story_df.head(10))
#print(story_df[story_df.channels.apply(lambda x: "Earnings" in x)]["body"].iloc[0])