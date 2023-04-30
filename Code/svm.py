#%%
import gzip
import json
import pandas as pd
import numpy as np
import gensim
import nltk
import statsmodels.graphics.api as smg
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from mord import OrdinalRidge, LogisticAT, LogisticIT
from pathlib import Path
from OrdClass import OrdClass
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


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

# %%
# word vectors
df = getDF(file, num_rows=300000)
# Grab reviewText and overall rating, drop NaNs
df = df[['reviewText', 'overall']]
df = df.dropna()
#%%
df['reviewLens'] = df['reviewText'].apply(lambda x: len(x.split()))
print(df['reviewLens'].describe())
# remove reviews less than 12 words
df = df[df['reviewLens'] >= 12]
# drop reviewLen
df = df.drop(columns=['reviewLens'])

# %%
# Load in BERT/NELA embeddings
def load_bert_nela(bert=True):
    data_path = Path('.') / 'data'
    if bert:
        raw_feats = pd.read_csv(data_path / "1000_raw_info.csv")
        data = pd.DataFrame(np.loadtxt(data_path / "1000_bert_feats.csv"))
        df = pd.concat([raw_feats, data], axis=1)
        df = df.drop(columns=["Unnamed: 0"])
        # rename stars col to overall
        df = df.rename(columns={"stars": "overall"})
        # rename rawText to reviewText
        df = df.rename(columns={"raw_text": "reviewText"})
        df.columns = ["feat_"+str(c) if c not in ['reviewText', 'overall'] else c for c in df.columns]
    else:
        raw_feats = pd.read_csv(data_path / "1000_raw_info.csv")
        data = pd.DataFrame(np.loadtxt(data_path / "1000_horne_feats.csv"))
        df = pd.concat([raw_feats, data], axis=1)
        df = df.drop(columns=["Unnamed: 0"])
        # rename stars col to overall
        df = df.rename(columns={"stars": "overall"})
        # rename rawText to reviewText
        df = df.rename(columns={"raw_text": "reviewText"})
        
    return df

df = load_bert_nela(bert=True)
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

#%%
# Export to csv
# df.to_csv(Path('.') / 'data' / 'glove_vectors_300k.csv', index=False)

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
# convert y values to start from 0 (0 -> 4) and convert to int from float
y_train_ord = (y_train - 1).astype(int)
y_test_ord = (y_test - 1).astype(int)

clf = OrdClass(SVC,clf_args={'kernel':'linear', 'probability':True})
clf.fit(X_train, y_train_ord)
y_pred_ord = clf.predict(X_test)
print('Ordinal classification report:\n', classification_report(y_test_ord, y_pred_ord))
# %%
# Multinomial log reg
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print('Logistic regression classification report:\n', classification_report(y_test, y_pred_logreg))
# %%
# Proportional odds log reg
def run(y, df, reg=True, grouped=False):
    formula = y + " ~ " + " + ".join(map(str, x)) # + " + " + bash_interaction(x[:4])
    mod = smf.mnlogit(formula=formula, data=df)
    res = None
    if reg:
        res = mod.fit_regularized(method="l1", disp=0)
    else:
        res = mod.fit(method="bfgs", maxiter=500, disp=25)
    # significant_vars = res.pvalues[res.pvalues < 0.05].index
    # if 'Intercept' in significant_vars:
    #     significant_vars = significant_vars.drop('Intercept')
    # percents = [(name, np.exp(res.params[name])-1) for name in significant_vars]
    # percents.sort(key = lambda x: abs(x[1]), reverse=True)
    # plt.title(f"Torndo Plot for {y}")
    # sns.barplot(x=[z[1] for z in percents], y=[z[0] for z in percents], orient='h')
    print("Psudo R2", res._results.prsquared)

    # print("LINK", link_test(res))
    # if grouped:
    #     print("PEARSON", standard_pearson(res, df))
    # else:
    #     print("HOSMER", Hosmer_Lemeshow(res))

    return res

def run_split(df, y_var):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    res = run(y_var, df_train, reg=False, grouped=False)
    yhat = res.predict(df_test)
    yhat = yhat.idxmax(axis=1)
    yhat = yhat + 1
    yhat = yhat.astype(int)
    y_test = df_test[y_var]
    print(classification_report(y_test, yhat))

x = df.drop(['reviewText', 'overall'], axis=1).columns
# x = df.drop(['feature_reviewText', 'feature_overall'], axis=1).columns
# make sure x is str
y = 'overall'
# y = 'feature_overall'
# res = run(y, df, reg=False)
run_split(df, y)
# %%
