# %%
import gzip
import json
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AdamW,AutoTokenizer
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from nela import NELAFeatureExtractor

import os
#os.chdir("..")


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




file = "Toys_and_Games_5.json.gz"
df = getDF(file)
filter_df = df[["overall", "reviewText"]].dropna()
text = filter_df.reviewText
y_vars = filter_df.overall

def horne_single(nela, t):
    try:
        feature_vector, _ = nela.extract_all(t)
    except:
        feature_vector = None
    
    return feature_vector

def bert_single(model, t):
    with torch.no_grad():
        try:
            inputs = tokenizer(t, return_tensors="pt").to(device)
            _, output = model(**inputs).to_tuple()
            out = output.cpu()
        except:
            out = None

    return out



target = 1000
star_count = {x:0 for x in range(1,6)}
stars = []
bert_feats = []
horne_feats = []
raw_text = []

nela = NELAFeatureExtractor()

device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

for i,(t,y) in enumerate(zip(text,y_vars)):
    if i % 1000 == 0:
       print(i)
       print(star_count)
    horne = horne_single(nela, t)
    bert = bert_single(model, t)
    if (bert is not None) and (horne is not None) and (star_count[y] < target):
        stars.append(y)
        bert_feats.append(bert)
        horne_feats.append(horne)
        raw_text.append(t)
        star_count[y]+=1
    if min(star_count.values()) == target:
        break

bert_feats = np.vstack(bert_feats)
horne_feats = np.vstack(horne_feats)
raw_info = pd.DataFrame({"stars":stars, "raw_text":raw_text})

bert_feats_name = f"{target}_bert_feats.csv"
horne_feats_name = f"{target}_horne_feats.csv"
raw_info_name = f"{target}_raw_info.csv"

np.savetxt(bert_feats_name, bert_feats)
np.savetxt(horne_feats_name, horne_feats)
raw_info.to_csv(raw_info_name)