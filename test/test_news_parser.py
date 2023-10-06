import pandas as pd
import pytest
from src.preprocessing import news_parser

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
    body = news_parser.remove_contact_info_sentences(text)
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
    short_name = news_parser.get_company_abbreviation(company_name, company_endings)    
    assert short_name == target_short_name


def test_parser():
    text = 'Cambium Networks Corporation ("Cambium Networks")'
    company_name="Cambium Networks Corporation"
    short_name="Cambium Networks"
    ticker=None
    body = news_parser.remove_company_specifics(body=text, 
                                                company_name=company_name, 
                                                short_name=short_name, 
                                                ticker=ticker)
    assert body == ' the company  (" the company ")'
    
def test_ticker_remover():
    text = "(NYSE:NVR)" 
    company_name=None
    short_name=None
    ticker="NVR"
    body = news_parser.remove_company_specifics(body=text, 
                                                company_name=company_name, 
                                                short_name=short_name, 
                                                ticker=ticker)
    assert body == ''