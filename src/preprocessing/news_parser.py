from benzinga import news_data, financial_data
import pandas as pd 
import json
import html2text
from bs4 import BeautifulSoup
import pickle
import itertools
from news_importer import RELEVANT_CHANNELS
import yfinance as yf
import re
from nltk.tokenize import sent_tokenize

def parse_story_to_row(story):
    # Converts JSON (dict) story to a list 
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


def filter_body(body, ticker, author):
    # Remove links
    # Identify all sentences with links (probably at the end of the document with links to company website with some advertisement...)
    # And remove them
    body = re.sub("www\.[a-z]*\.com", "REMOVE_THIS_SENTENCE", body) # remove sentences with links
    body = re.sub("[a-z]*@[a-z]*\.com", "REMOVE_THIS_SENTENCE", body) # remove sentences with emails
    body_dot_split = sent_tokenize(body)
    body = [sentence for sentence in body_dot_split if "REMOVE_THIS_SENTENCE" not in sentence]
    body = " ".join(body)

    # Get company name by ticker
    yf_ticker = yf.Ticker(ticker)
    try:
        company_name = yf_ticker.info['longName']
        # 2. Replace the name with "the company"
        company_endings = pd.read_table("data/corporation_endings.txt").iloc[:, 0]
        matching_mask = company_endings.apply(lambda x: x in company_name)
        longest_match_idx = company_endings[matching_mask].apply(lambda x: len(x)).idxmax()
        longest_match = company_endings.iloc[longest_match_idx]
        company_abbrev = company_name.replace(longest_match, "")
        body = body.replace(company_name, "the company").replace(company_abbrev, "the company")
    except:
        pass
    
    body = re.sub(f"\([A-Z]*:{ticker}\)", "REMOVE_THIS", body) # remove sentences with emails
    body = body.replace("REMOVE_THIS", "")

    # Remove Date 
    body = re.sub(" [A-Z][a-z]* [0-9][0-9], [0-9]* ", " ", body) # remove sentences with links
    
    # Remove author (preamble)
    body = re.sub(f".*\({author}\)", "", flags=re.IGNORECASE)
    return body


if __name__ == "__main__":
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

    story_df.loc[:, "body"] = story_df.apply(lambda x: filter_body(x.body, x.stocks, x.author), axis=1)

    print(story_df.shape)
    print(story_df.head(10))
    story_df.to_csv("data/stories.csv")
    #print(story_df[story_df.channels.apply(lambda x: "Earnings" in x)]["body"].iloc[0])