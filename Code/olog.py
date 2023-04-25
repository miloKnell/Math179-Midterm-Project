# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.graphics.api as smg
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.miscmodels.ordinal_model import OrderedModel
from collections import Counter
import os
import itertools
from scipy.stats.distributions import chi2
from scipy.stats.distributions import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
os.chdir(os.path.join("..", ".."))

#specifiy dataset, then read in appropriate data
data = "bert"

if data == "glove":
    df = pd.read_csv("glove_vectors.csv")
    df = df.drop(columns=["reviewText"])
    y = "overall"
    x = list(df.columns)
    x.remove(y)

elif data == "bert":
    raw_feats = pd.read_csv("100_raw_info.csv")
    data = pd.DataFrame(np.loadtxt("100_bert_feats.csv"))
    df = pd.concat([raw_feats, data], axis=1)
    df = df.drop(columns=["Unnamed: 0", "raw_text"])

    y = "stars"
    df.columns = ["feat_"+str(c) if c!= y else c for c in df.columns]
    x = list(df.columns)
    x.remove(y)
# %%
#run main regression on all x variables
def run(y, df):
    formula = y + " ~ " + " + ".join(x)
    mod = OrderedModel.from_formula(formula, data=df, distr="logit")
    res = mod.fit(method='bfgs', disp=False)
    return res

#train/test split
def run_split(df, y_var):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    res = run(y_var, df_train)
    yhat = res.predict(df_test)
    yhat = yhat.idxmax(axis=1)
    yhat = yhat + 1
    yhat = yhat.astype(int)
    y_test = df_test[y_var]
    print(classification_report(y_test, yhat))

run_split(df, y)