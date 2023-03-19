# %%
file = "Downloads/Electronics_5.json.gz"

import gzip
import json
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

# %%
from nela import NELAFeatureExtractor

import torch
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

text = ['This is the best novel I have read in 2 or 3 years.  It is everything that fiction should be -- beautifully written, engaging, well-plotted and structured.  It has several layers of meanings -- historical, family,  philosophical and more -- and blends them all skillfully and interestingly.  It makes the American grad student/writers\' workshop "my parents were  mean to me and then my professors were mean to me" trivia look  childish and silly by comparison, as they are.\nAnyone who says this is an  adolescent girl\'s coming of age story is trivializing it.  Ignore them.  Read this book if you love literature.\nI was particularly impressed with  this young author\'s grasp of the meaning and texture of the lost world of  French Algeria in the 1950\'s and \'60\'s...particularly poignant when read in  1999 from another ruined and abandoned French colony, amid the decaying  buildings of Phnom Penh...\nI hope the author will write many more books  and that her publishers will bring her first novel back into print -- I  want to read it.  Thank you, Ms. Messud, for writing such a wonderful work.', "HELLO WORLD"]

def horne(text):
    nela = NELAFeatureExtractor()
    out = []
    for t in text:
        feature_vector, _ = nela.extract_all(text[0])
        out.append(feature_vector)
    return np.array(out)


def bert(text):
    device = torch.device('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    out = []
    model.eval()
    with torch.no_grad():
        for t in text:
            inputs = tokenizer(t, return_tensors="pt").to(device)
            _, output = model(**inputs).to_tuple()
            out.append(output.cpu())

    out = np.vstack(out)
    return out

a,b = horne(text), bert(text)


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