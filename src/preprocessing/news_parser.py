import pickle
import re
import warnings

import datefinder
import html2text
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from dateutil.parser import UnknownTimezoneWarning
from nltk.tokenize import sent_tokenize

warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
import time

from sutime import SUTime


def get_company_abbreviation(company_name, company_endings):
    matching_mask = company_endings.apply(lambda x: x in company_name)
    if matching_mask.sum() == 0: return None
    longest_match_idx = company_endings[matching_mask].apply(lambda x: len(x)).idxmax()
    longest_match = company_endings.iloc[longest_match_idx]
    company_abbrev = company_name.replace(longest_match, "")
    return company_abbrev


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


def remove_date_specifics(body, pr_date):
    ## Habe unterschied. Datumsparser getestet: 
    # - dateparser, dateutil gefällt mir nicht
    # - datefinder ganz gut, allerdings teilweise zu aggressiv
    # - SUTime hat fast schon zu viel Funktionalität und braucht dementsprechend 
    # auch lange, ist aber als extra check vielleicht ganz gut, um zu überprüfen, 
    # ob die Datums von datefinder alle *vernünftig* sind.
    dates = datefinder.find_dates(body, source=True, strict=True)
    sutime = SUTime(mark_time_ranges=True, include_range=True)

    for date, text in dates:
        if len(sutime.parse(text)) != 1:
            print(f"SUTime didn't approve as a date: {text}")
            continue
        if pr_date.date() > date.date():
            body = body.replace(text, "past date")
        elif pr_date.date() == date.date():
            body = body.replace(text, "today")
        else:
            body = body.replace(text, "future date")
    return body


def remove_company_specifics(body, company_name, short_name, ticker):
    # Replace the actual company name with "the company"
    body = re.sub(f"({company_name})|({short_name})", " the company ", body)
    # Remove exchange/ticker info in parentheses
    body = re.sub(f"\([A-Z]*[ ]?:[ ]?{ticker}\)", "", body)
    return body


def remove_contact_info_sentences(body):
    # Remove links
    # Identify all sentences with links (probably at the end of the document with links to company website with some advertisement...)
    # And remove them

    LINK_SENTENCE_REGEX = "www\.[a-z]*\.com"
    EMAIL_SENTENCE_REGEX = "[a-z]*@[a-z]*\.com"
    PHONE_NUMBER_REGEX = "(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}"
    # Remove sentences with links
    # Remove sentences with emails
    f".([^.])*{LINK_SENTENCE_REGEX}([^.]*).?"
    
    body = re.sub(".([^.])*" + # End of some sentene, starting current sentence
                  "|".join([LINK_SENTENCE_REGEX, EMAIL_SENTENCE_REGEX, PHONE_NUMBER_REGEX]) + # regices of interest
                  "([^.]*).?", # End of current sentence.
                  ".", body) 
    return body


def filter_body(body, ticker, author, pr_date, company_name, short_name):
    body = remove_contact_info_sentences(body)
    body = remove_company_specifics(body, company_name, short_name, ticker)
    body = remove_date_specifics(body, pr_date)
    
    # Remove author (preamble)
    # TODO: ZUSAMMENFASSUNG KOMMT EVEL. VOR AUTHOR PRÄEMBEL, DANN IST es schlecht, alles vorher zu löschen
    body = re.sub(f"(\n.*{author})|(^.*{author})", "", body, flags=re.IGNORECASE) 
    
   # Remove (the "Company") parenthesis and other `("different name")`-constructs
    body = re.sub('\(.*"(.{1,})".*\)', "", body)

    # Remove underscores and dots
    body = re.sub("_|•", " ", body)
    
    # Remove weird symbols at the start of a new line (bullet points)
    remover_list = [
        "\*",
        "-",
        "\/",
        "\\\\",
    ]
    SYMBOLS_REGEX = "|".join(remover_list)
    body = re.sub(f"\n({SYMBOLS_REGEX})" + "{1,}", " ", body)

    # Remove weird symbol clusters
    body = re.sub(f"({SYMBOLS_REGEX})" + "{2,}", " ", body)

    # Remove underscors, exccess spaces, newlines and dots 
    remover_list = ["_", 
                    "( ){2,}",
                    "(\n){1,}",
                    "•"]
    body = re.sub("|".join(remover_list), " ", body)

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
