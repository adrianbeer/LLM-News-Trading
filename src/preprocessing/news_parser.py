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
import datefinder
from dateutil.parser import UnknownTimezoneWarning
import warnings
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)

with open("data/tickers.pkl",'rb') as f:
    TICKERS = pickle.load(f)


def filter_body(body, ticker, author, pr_date):
    # Remove links
    # Identify all sentences with links (probably at the end of the document with links to company website with some advertisement...)
    # And remove them

    LINK_SENTENCE_REGEX = "www\.[a-z]*\.com"
    EMAIL_SENTENCE_REGEX = "[a-z]*@[a-z]*\.com"
    PHONE_NUMBER_REGEX = "(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}"
    # Remove sentences with links
    # Remove sentences with emails
    body = re.sub("|".join([LINK_SENTENCE_REGEX, EMAIL_SENTENCE_REGEX, PHONE_NUMBER_REGEX]), "REMOVE_THIS_SENTENCE", body) 

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
    
    # Remove ticker info
    body = re.sub(f"\([A-Z]*[ ]?:[ ]?{ticker}\)", "REMOVE_THIS", body)
    body = body.replace("REMOVE_THIS", "")

    # Remove Dates
    dates = datefinder.find_dates(body, source=True, strict=True)
    for date, date_string in dates:
        if pr_date.date() < date.date():
            body = body.replace(date_string, "a past date")
        elif pr_date.date() == date.date():
            body = body.replace(date_string, "today")
        else:
            body = body.replace(date_string, "a future date")
    #body = re.sub(" [A-Z][a-z]* [0-9][0-9], [0-9]* ", " ", body)
    
    # Remove author (preamble)
    body = re.sub(f"(\n.*{author})|(^.*{author})", "\n", body, flags=re.IGNORECASE) # TODO: ZUSAMMENFASSUNG KOMMT EVEL. VOR AUTHOR PRÄEMBEL, DANN IST es schlecht, alles vorher zu löschen
    
    # Remove weird symbols
    SYMBOLS_REGEX = "(\)|-| |\/|\\\\|_)"
    body = re.sub(f"\n{SYMBOLS_REGEX}*|^({SYMBOLS_REGEX}|\")", "\n", body)
    body = body.replace("*", "")

    # Remove (the "Company") parenthesis 
    body = re.sub('(.*"Company".*)', "", body)
    body = body.strip("\n - \\")

    return body


if __name__ == "__main__":
    with open("data/story_df_raw.pkl", 'rb') as f:
        story_df = pickle.load(f)

    print(f"Filtered stories: {story_df.shape[0]}")

    story_df.loc[:, "body"] = story_df.apply(lambda x: filter_body(x.body, x.stocks, x.author, x.time), axis=1)

    print(story_df.shape)
    print(story_df.head(10))
    story_df.to_csv("data/stories.csv")
