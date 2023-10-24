from nltk import word_tokenize, pos_tag
import time
from nltk.stem.wordnet import WordNetLemmatizer
from typing import List
import numpy as np

### Generell -------------------------------

def lemmatize_words(string_list: List[str], wnl: WordNetLemmatizer) -> List[str]:
  # The try-except-block makes sure that the (lazy) corpus is properly loaded.
  # Sometimes the error `WordNetCorpusReade object has no attribute _LazyCorpusLoader__args`
  # occurrs, which is caught with the try-except-block
  def f(i, j):
    try:
      if j[0].lower() in ['a','n','v']:
        return wnl.lemmatize(i,j[0].lower()) 
      else:
          return wnl.lemmatize(i) 
    except AttributeError as e:
      print(e)
      time.sleep(1)
      return f(i, j)
  return [f(i, j) for i,j in pos_tag(string_list)]


def tokenize(doc):
    wnl = WordNetLemmatizer()

    # Tokenize document
    x = str.lower(doc)
    tokens = word_tokenize(x)

    # Remove words that are only one character.
    # Lemmatize the documents.
    lemmatized_tokens = np.array(lemmatize_words(tokens, wnl))
    return lemmatized_tokens


### TF-IDF ---------------------------------------


def keyword_filter(tokens: np.array, keyword_list: List[str]) -> np.array:
  # Tokens here are assumed to be lemmatized
  is_keyword_mask = [(x in keyword_list) for x in tokens]
  relevant_tokens = tokens[is_keyword_mask]
  return relevant_tokens