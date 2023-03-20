# %%
file = "Toys_and_Games_5.json.gz"

import gzip
import json
import pandas as pd
import numpy as np
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

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
    if i == 10000:
        break
  return pd.DataFrame.from_dict(df, orient='index')

david_df = getDF(file)
#%%
david_df = david_df[:10000]
# REMOVE: only use first 10k reviews
david_df = david_df[['reviewText', 'overall']]
david_df = david_df.dropna()
# %%
# import urllib.request
# import os

# # Download GloVe embeddings
# if not os.path.isfile('glove.6B.100d.txt'):
#     urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
# %%
# Lowercase the text
david_df['reviewText'] = david_df['reviewText'].str.lower()

# Tokenize the text
david_df['reviewText'] = david_df['reviewText'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
david_df['reviewText'] = david_df['reviewText'].apply(lambda x: [word for word in x if word not in stop_words])

# Join the tokens back into sentences
david_df['reviewText'] = david_df['reviewText'].apply(lambda x: ' '.join(x))
# %%
# Load GloVe embeddings
word_vectors = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False, no_header=True)

#%%
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
reviews = david_df['reviewText'].tolist()
# assert reviews has no nan
assert not any(pd.isnull(reviews))
review_vectors = get_avg_word_vectors(reviews, word_vectors)

# Create a new dataframe with the review vectors as features
df_vectors = pd.DataFrame(review_vectors)

# Rename columns
df_vectors.columns = ['feature_' + str(col) for col in df_vectors.columns]

# Concatenate with original dataframe
df_final = pd.concat([david_df, df_vectors], axis=1)

# drop nan
df_final = df_final.dropna()
# export review vectors
df_final.to_csv('review_vectors.csv')

# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objs as go


# Create a subset of the dataframe for good reviews (4/5 stars) and bad reviews (1/2 stars)
df_good = df_final[df_final['overall'].isin([4, 5])]
df_bad = df_final[df_final['overall'].isin([1, 2])]

# Concatenate the good and bad reviews dataframes
df_subset = pd.concat([df_good, df_bad], axis=0)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df_subset.iloc[:, 11:])

# Create a scatter plot of the t-SNE results, separating good and bad reviews by color
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:len(df_good), 0], tsne_results[:len(df_good), 1], c='blue', label='Good Reviews')
plt.scatter(tsne_results[len(df_good):, 0], tsne_results[len(df_good):, 1], c='red', label='Bad Reviews')
plt.legend()
plt.title('t-SNE Visualization of GloVe Embeddings')
plt.show()
# %%
from sklearn.decomposition import PCA

# Perform PCA to reduce the dimensionality of the word embeddings
pca = PCA(n_components=3)
embedding_pca = pca.fit_transform(review_vectors)

# Get the star ratings for each review
ratings = david_df['overall'].tolist()

# Create a scatter plot of the embeddings, with different colors for each rating
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(1, 6):
    plt.scatter(embedding_pca[np.array(ratings) == i][:, 0], embedding_pca[np.array(ratings) == i][:, 1], c=colors[i-1], label=f'{i} stars')
plt.legend()
plt.show()

# %%
good_vectors = [v for v, r in zip(review_vectors, ratings) if r >= 4]
bad_vectors = [v for v, r in zip(review_vectors, ratings) if r <= 2]

pca = PCA(n_components=3)
good_pca = pca.fit_transform(good_vectors)
bad_pca = pca.fit_transform(bad_vectors)

# Create a scatter plot of the embeddings, with different colors for good and bad reviews
plt.figure(figsize=(10, 8))
plt.scatter(good_pca[:, 0], good_pca[:, 1], c='blue', label='Good Reviews')
plt.scatter(bad_pca[:, 0], bad_pca[:, 1], c='red', label='Bad Reviews')
plt.title('PCA Visualization of GloVe Embeddings')
plt.legend()
plt.show()
# %%
