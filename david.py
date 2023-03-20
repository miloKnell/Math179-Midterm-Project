# %%
file = "Toys_and_Games_5.json.gz"

import gzip
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#%% 


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF(file)
# %%
import urllib.request
import os

# Download GloVe embeddings
if not os.path.isfile('glove.6B.100d.txt'):
    urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
# %%
# Lowercase the text
df['reviewText'] = df['reviewText'].str.lower()

# Tokenize the text
df['reviewText'] = df['reviewText'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['reviewText'] = df['reviewText'].apply(lambda x: [word for word in x if word not in stop_words])

# Join the tokens back into sentences
df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join(x))
# %%
