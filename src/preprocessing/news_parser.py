import pandas as pd 
import html2text
from bs4 import BeautifulSoup
import pickle
import html2text
from bs4 import BeautifulSoup
import yfinance as yf
import re
from nltk.tokenize import sent_tokenize
import datefinder
from dateutil.parser import UnknownTimezoneWarning
import warnings
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
import time

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


def filter_body(body, ticker, author, pr_date):
    body = body_formatter(body)
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
    
    # Remove exchange/ticker info
    body = re.sub(f"\([A-Z]*[ ]?:[ ]?{ticker}\)", "REMOVE_THIS", body)
    body = body.replace("REMOVE_THIS", "")

    # Remove Dates
    dates = datefinder.find_dates(body, source=True, strict=True)
    for date, date_string in dates:
        if pr_date.date() > date.date():
            body = body.replace(date_string, "a past date")
        elif pr_date.date() == date.date():
            body = body.replace(date_string, "today")
        else:
            body = body.replace(date_string, "a future date")
    
    # Remove author (preamble)
    body = re.sub(f"(\n.*{author})|(^.*{author})", "", body, flags=re.IGNORECASE) # TODO: ZUSAMMENFASSUNG KOMMT EVEL. VOR AUTHOR PRÄEMBEL, DANN IST es schlecht, alles vorher zu löschen
    
   # Remove (the "Company") parenthesis and other `("different name")`-constructs
    body = re.sub('\(.*"(.{1,})".*\)', "", body)

    # Remove underscores
    body = body.replace("_", " ")
    
    # Remove bullet point dots
    body = body.replace("•", " ")

    # Remove weird symbols at the start of a new line (bullet points)
    SYMBOLS_REGEX = "(\*|-|\/|\\\\)"
    body = re.sub(f"\n{SYMBOLS_REGEX}"+ "{1,}", " ", body)

    # Remove weird symbol clusters
    body = re.sub(f"{SYMBOLS_REGEX}"+ "{2,}", " ", body)

    # Replace all new lines with a space
    body = body.replace("\n", " ")

      # Remove exccess spaces
    body = re.sub("( ){2,}", " ", body)

    # Remove exccess spaces
    body = re.sub("( ){2,}", " ", body)

    # Remove exccess dots
    body = re.sub("(( )*\.){1,}", ".", body)


    # Final stripping of stuff at the start/end of the file
    body = body.strip("\n -\\_*/().'")

    return body


if __name__ == "__main__":
    with open("data/story_df_raw.pkl", 'rb') as f:
        story_df = pickle.load(f)

    print(f"Filtered stories: {story_df.shape[0]}")

    start = time.time()
    story_df.loc[:, "body"] = story_df.apply(lambda x: filter_body(x.body, x.stocks, x.author, x.time), axis=1)
    end = time.time()
    print(f"Time elapsed: {end-start}s")
    print(f"Average seconds required per body: {(end-start)/story_df.shape[0]}s")

    story_df.loc[:, "NewsTimestamp"] = pd.to_datetime(story_df.time)
    story_df.drop(columns=["time"], inplace=True)

    print(story_df.shape)
    print(story_df.head(10))

    story_df.to_pickle("data/stories.pkl")
    story_df.to_csv("data/stories.csv")
