from datetime import timedelta
import pandas as pd
import pytest
from src.utils.preprocessing import news_preprocessing

contact_info_test_sentences = [
    ("Hey. Ho. Now comes an E-Mail: www.website.com. Now the Text continues.", 
     "Hey. Ho. Now the Text continues."),
    ("Hey. Check out www.nvrinc.com, www.ryanhomes.com, www.nvhomes.com and www.heartlandluxuryhomes.com. Now the Text continues.", 
     "Hey. Now the Text continues."),
    ("Hey. View original content:https://www.prnewswire.com/news-releases/nvr-inc-\nannounces-share-repurchase-301348550.html\n\nSOURCE NVR, Inc.", 
    "Hey."),
    ("Hey. View original content:https://www.prnewswire.com/news-releases/nvr-inc-\nannounces-share-repurchase-301348550.html\n\nSOURCE NVR, Inc", 
     "Hey."),
    ("For more info visit www.HanesForGood.com.",
     ""),
    ("Visit at https: newsroom.hanesbrands.com/.",
     "")
]
@pytest.mark.parametrize("text, output", contact_info_test_sentences)
def test_remove_contact_info_sentences(text, output):
    body = news_preprocessing.remove_contact_info_sentences(text)
    assert body == output
    
    
company_name_tests = [
    ("GeneDx Holdings Corp.", "GeneDx"),
    ("The Boeing Company", "Boeing"),
    ("Verizon Communications Inc.", "Verizon Communications"),
    ("Masimo Corporation", "Masimo"),
]
@pytest.mark.parametrize("company_name, target_short_name", company_name_tests)
def test_get_company_abbreviation(company_name, target_short_name):
    company_endings = pd.read_table("data_shared/corporation_endings.txt").iloc[:, 0]
    short_name = news_preprocessing.get_company_abbreviation(company_name, company_endings)    
    assert short_name == target_short_name


def test_parser():
    text = 'Cambium Networks Corporation ("Cambium Networks")'
    company_name="Cambium Networks Corporation"
    short_name="Cambium Networks"
    ticker=None
    body = news_preprocessing.remove_company_specifics(body=text, 
                                                company_name=company_name, 
                                                short_name=short_name, 
                                                ticker=ticker)
    assert body == ' the company  (" the company ")'
    
def test_ticker_remover():
    text = "(NYSE:NVR)" 
    company_name=None
    short_name=None
    ticker="NVR"
    body = news_preprocessing.remove_company_specifics(body=text, 
                                                company_name=company_name, 
                                                short_name=short_name, 
                                                ticker=ticker)
    assert body == ''
    
    
def test_filter_body():
    text = 'MAYFIELD VILLAGE, Ohio--(BUSINESS WIRE)--\n\nAccording to digital measurement company comScore, Inc., one in four mobile\nusers now have smart phones and by this time next year, research company\nNielsen estimates the number will be one in two. To meet the growing demand\nfor information on-the-go, the Progressive car insurance group is expanding\nits mobile offerings.\n\nProgressive\'s new "Flo-isms" app includes 48 popular Flo sound bites from\nProgressive\'s commercials. The app, featuring Flo\'s voice, is available on the\niPhone, iPad, iTouch and Google Android(TM) operating system. (Photo: Business\nWire)\n\n“We\'ve improved our core car insurance apps to bring even more useful tools to'
    company_name=None
    short_name=None
    ticker="SCOR"
    row = pd.Series({
        "body":text,
        "stocks":ticker, 
        "author":"BUSINESS WIRE", 
        "time": pd.Timestamp(year=2000, month=1, day=1), 
        "company_name":"comScore, Inc.", 
        "short_name":"comScore"
    })
    body = news_preprocessing.filter_body(row)
    assert body == '“We\'ve improved our core car insurance apps to bring even more useful tools to'

def test_is_feasible_date():
    d = timedelta(hours=1)
    assert news_preprocessing.is_feasible_date(d) is False 