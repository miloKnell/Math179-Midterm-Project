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


#functions for parsing and loading df. provided by the owner of the data repo
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

#functions to extract the featurizations for a single review, given a text t and the required class
#note the try/accept block, sometimes the code fails. For horne it is because the researchers who wrote the functions had some bugs
#for bert it is when the tokenizer has bugs -- to my understanding due to malformatted data, like empty / not long enough / too long. 
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


#target is number of reviews per class to take (so if target=1000 this will result in a dataset with n=5000)
target = 1000
star_count = {x:0 for x in range(1,6)}
stars = []
bert_feats = []
horne_feats = []
raw_text = []

nela = NELAFeatureExtractor()

#can be changed to cuda, but is faster for non-batched 
device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #currently using uncasd -- future work might make this cased
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

#main loop, run through data until we either run out (which is hard with our millions of points) or hit target for each catagory
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

#combine the features and save to disk
bert_feats = np.vstack(bert_feats)
horne_feats = np.vstack(horne_feats)
raw_info = pd.DataFrame({"stars":stars, "raw_text":raw_text})

bert_feats_name = f"{target}_bert_feats.csv"
horne_feats_name = f"{target}_horne_feats.csv"
raw_info_name = f"{target}_raw_info.csv"

np.savetxt(bert_feats_name, bert_feats)
np.savetxt(horne_feats_name, horne_feats)
raw_info.to_csv(raw_info_name)