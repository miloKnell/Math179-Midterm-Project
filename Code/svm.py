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
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
nltk.download('punkt')
nltk.download('stopwords')


file = Path('..') / "Toys_and_Games_5.json.gz"
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

df = getDF(file, num_rows=1000)
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
# Load GloVe embeddings
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(Path('..') / 'glove.6B.100d.txt', binary=False, no_header=True)
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
df_final = pd.concat([df, df_vectors], axis=1)
# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_final.drop(['reviewText', 'overall'], axis=1), df_final['overall'], test_size=0.2, random_state=42)

# Create an instance of the SVM classifier
svm = SVC(kernel='linear', decision_function_shape='ovo')

# Train the SVM on the training data
svm.fit(X_train, y_train)

# Test the SVM on the testing data
y_pred = svm.predict(X_test)

# Evaluate the performance of the SVM
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# %%
