#%%
import gzip
import json
import pandas as pd
import numpy as np
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from mord import OrdinalRidge
from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')


file = Path('..') / "Toys_and_Games_5.json.gz"
# %%
# Load GloVe embeddings
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(Path('..') / 'glove.6B.100d.txt', binary=False, no_header=True)
# %%
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


def getDF(path, num_rows=None):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if num_rows is not None and i == num_rows:
            break
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF(file, num_rows=100000)
# %%
# Grab reviewText and overall rating, drop NaNs
df = df[['reviewText', 'overall']]
df = df.dropna()

#%%
# # Undersample 4, 5 star reviews -> more balanced
# # Make sum of 4, 5 star reviews equal to sum of 1, 2, 3 star reviews
# df_majority = df[df.overall.isin([4,5])]
# df_minority = df[df.overall.isin([1,2,3])]

# df_majority = resample(df_majority,
#                                     replace=False,    # sample without replacement
#                                     n_samples=len(df_minority),     # to match minority class
#                                     random_state=42) # reproducible results

# df = pd.concat([df_majority, df_minority])
# # plot bar chat of value_counts, sort reviews by rating
# df['overall'].value_counts().sort_index().plot(kind='bar')

#%%
# Undersample all reviews to be equal to the one with the least data
min_count = df['overall'].value_counts().min()

df_balanced = pd.DataFrame()

for rating in df['overall'].unique():
    df_rating = df[df['overall'] == rating]
    df_rating_downsampled = resample(df_rating, replace=False, n_samples=min_count, random_state=42)
    df_balanced = pd.concat([df_balanced, df_rating_downsampled])

df = df_balanced.reset_index(drop=True)
# plot bar chat of value_counts, sort reviews by rating
# df['overall'].value_counts().sort_index().plot(kind='bar')
print(df['overall'].value_counts().sort_index())
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
# Define a function to calculate the average word vector for each review
def get_avg_word_vectors(reviews, word_vectors):
    """
    Given a list of reviews and word vectors, returns a list of average word vectors for each review
    """
    review_vectors = []
    for review in reviews:
        words = review.split()
        word_vectors_list = [word_vectors.get_vector(word) for word in words if word in word_vectors.key_to_index]
        if len(word_vectors_list) > 0:
            review_vectors.append(np.mean(word_vectors_list, axis=0))
        else:
            review_vectors.append(np.zeros(word_vectors.vector_size))
    return review_vectors
  
# Calculate the average word vectors for each review
reviews = df['reviewText'].tolist()
# assert reviews has no nan
assert not any(pd.isnull(reviews))
review_vectors = get_avg_word_vectors(reviews, word_vectors)

# Create a new dataframe with the review vectors as features
df_vectors = pd.DataFrame(review_vectors)

# Rename columns
df_vectors.columns = [f'feature_{str(col)}' for col in df_vectors.columns]

# Concatenate with original dataframe
df = pd.concat([df, df_vectors], axis=1)
# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['reviewText', 'overall'], axis=1), df['overall'], test_size=0.2, random_state=42)

# %%
# Train and test an OVR SVM classifier
ovr_clf = OneVsRestClassifier(SVC(kernel='linear'))
ovr_clf.fit(X_train, y_train)
y_pred_ovr = ovr_clf.predict(X_test)
print('OVR classification report:\n', classification_report(y_test, y_pred_ovr))

#%%
# Train and test an OVO SVM classifier
ovo_clf = OneVsOneClassifier(SVC(kernel='linear'))
ovo_clf.fit(X_train, y_train)
y_pred_ovo = ovo_clf.predict(X_test)
print('OVO classification report:\n', classification_report(y_test, y_pred_ovo))
# %%
# Mord OrdinalRidge
ord_reg = OrdinalRidge()
ord_reg.fit(X_train, y_train)

y_pred_ord = ord_reg.predict(X_test)

print('Mord OrdinalRidge classification report:\n', classification_report(y_test, y_pred_ord))

# %%
from sklearn.base import BaseEstimator
class OrdClass(BaseEstimator):
  """
  Helper class that solves ordinal classification (classes that have an order to them eg cold,warm,hot)
  """
  def __init__(self,classifier=None,clf_args=None):
    """
    y needs to be a number that start from 0 and increments by 1
    classifier object needs to be able to return a probability
    """
    self.classifier = classifier
    self.clfs = []
    self.clf_args = clf_args
    self.final_prob = None
  
  def fit(self,X,y,**fit):
    self.X = X
    self.y = y
    import copy
    no_of_classifiers = np.max(self.y) #since y starts from 0
    self.classes_ = list(range(no_of_classifiers+1))
    if isinstance(self.clf_args,list):
      #for pipelines
      c = self.classifier(self.clf_args)
    elif isinstance(self.clf_args,dict):
      #for normal estimators
       c = self.classifier(**self.clf_args)
    for i in range(no_of_classifiers):
      # make a copy of y because we want to change the values of y
      copy_y = np.copy(self.y)
      # make a binary classification here
      copy_y[copy_y<=i] = 0
      copy_y[copy_y>i] = 1
      classifier = copy.deepcopy(c)
      classifier.fit(self.X,copy_y,**fit)
      self.clfs.append(classifier)
    return self
  def predict_proba(self,test):
    prob_list = []
    final_prob = []
    length = len(self.clfs)
    for clf in self.clfs:
      prob_list.append(clf.predict_proba(test)[:,1])
    for i in range(length+1):
      if i == 0:
        final_prob.append(1-prob_list[i])
      elif i == length:
        final_prob.append(prob_list[i-1])
      else:
        final_prob.append(prob_list[i-1]-prob_list[i])
    answer = np.array(final_prob).transpose()
    self.final_prob= answer
    return answer
  def predict(self,test):
    self.predict_proba(test)
    return np.argmax(self.final_prob,axis=1)
  def score(self,X,y,sample_weight=None):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
# %%
# convert y values to start from 0 (0 -> 4) and convert to int from float
y_train = (y_train - 1).astype(int)
y_test = (y_test - 1).astype(int)

clf = OrdClass(SVC,clf_args={'kernel':'linear', 'probability':True})
clf.fit(X_train,y_train)
y_pred_ord = clf.predict(X_test)
print('Ordinal classification report:\n', classification_report(y_test, y_pred_ord))
# %%
