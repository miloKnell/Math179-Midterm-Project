# %%
<<<<<<< Updated upstream
#file = "Downloads/Electronics_5.json.gz"

#import gzip
#import json
#def parse(path):
#  g = gzip.open(path, 'r')
#  for l in g:
#    yield json.loads(l)
=======
file = "Toys_and_Games_5.json.gz"

import gzip
import json
import pandas as pd


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)
>>>>>>> Stashed changes


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
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

filter_df = df[["overall", "reviewText"]].dropna()
text = filter_df.reviewText[:10**5]

# def horne(text):
# nela = NELAFeatureExtractor()
# out = []
# for i,t in enumerate(text):
#     if i % 1000 == 0:
#        print(i)
#     try:
#         feature_vector, _ = nela.extract_all(t)
#         out.append(feature_vector)
#     except:
#        print("HIT ERROR ON", i)
# out = np.array(out)


# def bert(text):
device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device)

out = []
model.eval()
with torch.no_grad():
    for i,t in enumerate(text):
        if i % 1000 == 0:
            print(i)
        inputs = tokenizer(t, return_tensors="pt").to(device)
        _, output = model(**inputs).to_tuple()
        out.append(output.cpu())

out = np.vstack(out)
    return out

#horne_feats = horne(df.reviewText)
#bert_feats = bert(df.reviewText)


# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

scalar = StandardScaler()
x_trans = scalar.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.2, random_state=42)

svc = LinearSVC(C=1)

svc.fit(X_train, y_train)
pred = svc.predict(X_test)
acc = sum(pred==y_test)/len(pred)
