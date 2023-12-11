import pickle
import re
from typing import List
import warnings

import html2text
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from dateutil.parser import UnknownTimezoneWarning

warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
import time

from sutime import SUTime
import datefinder
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers


# These constants are used in `remove_date_specifics`
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PARSERS = [parser for parser in default_parsers if parser != 'relative-time']


# Used in remove contact info sentences
ACRONYMS = r'(?:[A-Z]\.)+'
LINK_SENTENCE_REGEX = "[a-z]*\.[A-za-z]*\.com"
EMAIL_SENTENCE_REGEX = "[A-za-z]*@[A-za-z]*\.com"
PHONE_NUMBER_REGEX = "(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}"
HTML_LINK_INDEX = "\.html"
CONTACT_INFO_REGEX = "|".join([LINK_SENTENCE_REGEX, 
                                EMAIL_SENTENCE_REGEX, 
                                PHONE_NUMBER_REGEX, 
                                HTML_LINK_INDEX])


def infer_author(body):
  for author in ["PRNewswire", "Globe Newswire", "Business Wire", "ACCESSWIRE"]:
    if re.search(author, body, re.IGNORECASE) is not None:
      return author
  return None


# Get company name by ticker (longName is always equal to shortName in yf...)
# This takes a long time, because of the api calls to yf ~30min?
def yahoo_get_wrapper(x):
  try:
    return yf.Ticker(x).info.get("longName")
  except:
    return None


def get_company_abbreviation(company_name, company_endings):
    f = lambda x: _get_company_abbreviation(x, company_endings=company_endings) if x else x
    return f(f(company_name))


def _get_company_abbreviation(company_name, company_endings):
    # Special treatment for double-sided name wrappers
    if re.search("The (.)* Company", company_name):
        return re.sub("(The )|( Company)", "", company_name)  
    
    matching_mask = company_endings.apply(lambda x: company_name.endswith(x))
    if matching_mask.sum() == 0: return company_name
    first_match = company_endings[matching_mask].iloc[0]
    company_abbrev = company_name.replace(first_match, "")
    return company_abbrev.strip(" ")


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

def remove_date_specifics(body, pr_date, logging=False):
    ## Habe unterschied. Datumsparser getestet: 
    # - dateparser, dateutil gefällt mir nicht
    # - datefinder ganz gut, allerdings teilweise zu aggressiv
    # - SUTime hat fast schon zu viel Funktionalität und braucht dementsprechend 
    # auch lange, ist aber als extra check vielleicht ganz gut, um zu überprüfen, 
    # ob die Datums von datefinder alle *vernünftig* sind.
    datefinder_dates = datefinder.find_dates(body, 
                                             source=True, 
                                             strict=True)
    
    dateparser_dates = search_dates(body, 
                                    languages=["en"], 
                                    settings={"STRICT_PARSING":True,
                                              "PARSERS": PARSERS})
    if dateparser_dates is None:
        dateparser_dates = []
    dateparser_dates = [(date, text) for text, date in dateparser_dates]
    
    sutime = SUTime(mark_time_ranges=True, include_range=True)
    contains_month_or_day = lambda x: bool(re.search("|".join(MONTHS + DAYS), x))
    
    for date, text in list(datefinder_dates) + dateparser_dates:
        if (len(sutime.parse(text)) != 1) and not contains_month_or_day(text):
            if logging: print(f"Not approved as a date: {text}")
            continue
        if logging: print(f"Approved as date {text}")
        body = body.replace(text, "")
    return body


def remove_company_specifics(body, company_name, short_name, ticker):
    # Remove exchange/ticker info in parentheses
    # Doing this first is important if ticker equals short_name
    body = re.sub(f"\([A-Za-z ]*:[ ]*{ticker}\)", "", body)
    
    # Replace the actual company name with "the company"
    body = re.sub(f"(\*\*)*{company_name}(\*\*)*", " the company ", body)
    # The full name should take precedence, otherwise trailing ", LLC" for example.
    body = re.sub(f"(\*\*)*({short_name})(\*\*)*", " the company ", body)

    return body


def remove_contact_info_sentences(body):
    # Note: If we want to remove numbers, make sure to do this last, because e.g. Tickers can contain numbers...
    
    # Remove links
    # Identify all sentences with links (probably at the end of the document with links to company website with some advertisement...)
    # And remove them
    # Remove sentences with links
    # Remove sentences with emails

    # This way is required, in case of multiple web links in a sentence for example
    body = re.sub(CONTACT_INFO_REGEX, 
                  "REMOVESENTENCE", 
                  body)
    body = re.sub("[^\.]*" + # End of some sentene, starting current sentence
                  f"(REMOVESENTENCE)" + # regices of interest
                  f"([^\.]|{ACRONYMS})*\.?", # End of current sentence.
                  "", 
                  body) 
    return body


def remove_patterns(patterns: List[str], remove_with: str, text:str, flags=None):
  mask = bytearray(len(text))

  for pattern in patterns:
    for match in re.finditer(pattern, text, flags):
      r = range(match.start(), match.end())

      mask[r] = 'x' * len(r)

  return remove_with.join(character for character, bit in zip(text, mask) if not bit)


def filter_body(row: pd.Series, logging=False) -> str:
    '''
    TODO: Vor dem parsen sollten wir den Titel noch vorne anfügen, falls dort der Unternehmensname auch vorkommt.
    Das werden wir ja sowieso tun...
    '''
    body, ticker, author, pr_date, company_name, short_name = row.body, row.stocks, row.author, row.time, row.company_name, row.short_name
    
    # Remove newline symbols which can interfere
    # with the date detection process
    body = re.sub("(\n){1,}", 
                  " ", 
                  body)
    
    body = remove_company_specifics(body, company_name, short_name, ticker)
    body = remove_contact_info_sentences(body)
    body = remove_date_specifics(body, pr_date, logging=logging)
    
    
    # TODO: ZUSAMMENFASSUNG KOMMT EVEL. VOR AUTHOR PRÄEMBEL, DANN IST es schlecht, alles vorher zu löschen
    body = remove_patterns([f"(.*{author})|(^.*{author})", # Remove author (preamble)
                            '\(.*"(.{1,})".*\)'], # Remove (the "Company") parenthesis and other `("different name")`-constructs
                           "", 
                           body,
                           flags=re.IGNORECASE)
    
    # Converts itemized lists to sentences.
    body = re.sub("(\*|•)", 
                  ".", 
                  body)
    
    # Remove weird symbol (clusters)
    remover_list = [
        "\*",
        "-",
        "\/",
        "\\\\",
    ]
    remove_regex1 = "|".join(remover_list) + "{1,}"

    # Remove underscors, exccess spaces, newlines and dots 
    remover_list = ["_", 
                    "( ){2,}",
                    "•"]
    remove_regex2 = "|".join(remover_list)

    body = remove_patterns([remove_regex1, 
                            remove_regex2],
                            " ", 
                            body)


    # Remove exccess dots and white space around them
    body = re.sub("(( )*(\.)( )*){1,}", ".", body)

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
