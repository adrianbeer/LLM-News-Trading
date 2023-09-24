from src.preprocessing import news_parser


def test_remove_contact_info_sentences():
    text = "Hey. Ho. Now comes an E-Mail: www.website.com. Now the Text continues."
    body = news_parser.remove_contact_info_sentences(text)
    assert body == "Hey. Ho. Now the Text continues."
