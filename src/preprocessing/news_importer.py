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

no_error = True
i = 1
while no_error and i <= 100:
    try:
        #  channel="Earnings"
        stories = paper.news(display_output="full", page=i, pagesize=100, )
        all_stories.append(stories)
        i += 1
    except: 
        no_error = False

with open("data/stories_raw.pkl", "wb") as f:
    pickle.dump(all_stories, f)
