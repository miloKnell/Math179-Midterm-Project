# %%
file = "Toys_and_Games_5.json.gz"

import gzip
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


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
from nela import NELAFeatureExtractor

import torch
from transformers import BertModel, BertTokenizer, AdamW,AutoTokenizer
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader


filter_df = df[["overall", "reviewText"]].dropna()
n = 10**2
text = filter_df.reviewText[:n]
y_vars = filter_df.overall[:n]

# def horne(text):
nela = NELAFeatureExtractor()
out = []
y = []
for i,(t,y_indivudial) in enumerate(zip(text,y_vars)):
    if i % 1000 == 0:
       print(i)
    try:
        feature_vector, _ = nela.extract_all(t)
        out.append(feature_vector)
        y.append(y_indivudial)
    except:
       print("HIT ERROR ON", i)
out = np.array(out)


# def bert(text):
# device = torch.device('cpu')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased").to(device)

# out = []
# model.eval()
# y = []
# with torch.no_grad():
#     for i,(t,y_indivudial) in enumerate(zip(text,y_vars)):
#         if i % 1000 == 0:
#             print(i)
#         try:
#           inputs = tokenizer(t, return_tensors="pt").to(device)
#           _, output = model(**inputs).to_tuple()
#           out.append(output.cpu())
#           y.append(y_indivudial)
#         except:
#            print("FAILED", i)

# out = np.vstack(out)
#horne_feats = horne(df.reviewText)
#bert_feats = bert(df.reviewText)


# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

scalar = StandardScaler()
x_trans = scalar.fit_transform(out)
X_train, X_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.2, random_state=42)

svc = LinearSVC(C=1)

svc.fit(X_train, y_train)
pred = svc.predict(X_test)
acc = sum(pred==y_test)/len(pred)
