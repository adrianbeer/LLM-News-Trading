from benzinga import news_data, financial_data
import pandas as pd 
import json

import time

api_key = "272764ec2a3b4c3188dbef1310a699e7"
fin = financial_data.Benzinga(api_key)
paper = news_data.News(api_key)


def parse_story_to_row(story):
    # Converts JSON (dict) story to a list 
    tags: list  = [d["name"] for d in story["tags"]]
    channels: list = [d["name"] for d in story["channels"]]

    time = pd.to_datetime(story["created"])
    stocks = [d["name"] for d in story["stocks"]]

    if len(stocks) != 1: 
        return None
    
    ticker = stocks[0] # This is the ticker(s)
    html_body = story["body"]
    title = story["title"]
    author = story["author"]
    id = story["id"]

    return id, title, tags, json.dumps(channels), ticker, html_body, time, author


def save_df_and_return_empty(story_df: pd.DataFrame, current_year):
    print(f"#rows: {story_df.shape[0]}")
    story_df.to_parquet(path=f"data/parquet_bzg_stories/story_df_raw_{current_year}.parquet")
    story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "html_body"], dtype=object)
    return story_df


if __name__ == "__main__":
    # Initialize empty data frame
    story_df = pd.DataFrame(columns=["time", "stocks", "author", "title", "channels", "html_body"], dtype=object)
    
    # 100 is the max pagesize.
    pagesize = 100

    current_year = 0

    # Before 2017 there are no channels!
    ranger_time = pd.date_range(start="2023-01-01", end="2023-09-01", freq="D") 
    ranger = [x.strftime("%Y-%m-%d") for x in ranger_time]

    for r in range(len(ranger)):
        i = 0
        no_error = 0
        while no_error <= 2:
            try:
                time.sleep(0.6)
                print(i)
                stories = paper.news(display_output="full", page=i, date_from=ranger[r], date_to=ranger[r+1], pagesize=100)

                print(f"Number of stories {len(stories)}")
                if len(stories) == 0: 
                    no_error += 1
                for s in range(len(stories)):
                    story = stories[s]
                    parsed_story = parse_story_to_row(story)

                    if parsed_story is None: continue
                    id, title, tags, channels, stocks, html_body, timestamp, author = parsed_story

                    year = timestamp.year

                    if current_year != year:
                        story_df = save_df_and_return_empty(story_df, current_year)
                        current_year = year
                        print(year)

                    story_df.loc[id, ["time", "stocks", "author", "title", "channels", "html_body"]] = timestamp, stocks, author, title, channels, html_body

                i += 1
            except Exception as e:
                print(e) 
                no_error += 1

    save_df_and_return_empty(story_df, current_year)

    



