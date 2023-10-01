from src.preprocessing import news_parser


def test_remove_contact_info_sentences():
    text = "Hey. Ho. Now comes an E-Mail: www.website.com. Now the Text continues."
    body = news_parser.remove_contact_info_sentences(text)
    assert body == "Hey. Ho. Now the Text continues."

    # Multiple links in a sentence
    text = "Hey. Check out www.nvrinc.com, www.ryanhomes.com, www.nvhomes.com and www.heartlandluxuryhomes.com. Now the Text continues."
    body = news_parser.remove_contact_info_sentences(text)
    assert body == "Hey. Now the Text continues."
    
    text = "Hey. View original content:https://www.prnewswire.com/news-releases/nvr-inc-\nannounces-share-repurchase-301348550.html\n\nSOURCE NVR, Inc."
    body = news_parser.remove_contact_info_sentences(text)
    assert body == "Hey."
    
    # No dot at the end
    "Hey. View original content:https://www.prnewswire.com/news-releases/nvr-inc-\nannounces-share-repurchase-301348550.html\n\nSOURCE NVR, Inc"
    body = news_parser.remove_contact_info_sentences(text)
    assert body == "Hey."
    
    
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