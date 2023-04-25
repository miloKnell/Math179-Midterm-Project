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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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
#these are a bunch of helper functions to ensure our regression is statistically valid
def corrmat(df):
    corr_mat = np.corrcoef(df.T)
    plt.rc("figure", figsize=(18, 10))
    smg.plot_corr(corr_mat, xnames=df.columns)

def VIF(df):
    vif = pd.DataFrame()
    X = add_constant(df)
    vif["VIF Factor"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    print(vif.round(1))

#General: UCLA https://stats.oarc.ucla.edu/stata/webbooks/logistic/chapter3/lesson-3-logistic-regression-diagnostics/
#General: Hosmer and Lemeshow https://ftp.idu.ac.id/wp-content/uploads/ebook/ip/REGRESI%20LOGISTIK/epdf.pub_applied-logistic-regression-wiley-series-in-probab.pdf 



#https://www.jstor.org/stable/2346405?origin=crossref
def link_test(res):
    data = {"hat":res.fittedvalues}
    data["hatsq"] = data["hat"]**2
    data["y"] = res.model.endog
    mod_2 = smf.logit(formula="y ~ hat + hatsq", data=data)
    res_2 = mod_2.fit(disp=0)
    p = res_2.pvalues["hatsq"]
    test_stat = norm.ppf(1-p/2)
    return p, test_stat


#http://www.medicine.mcgill.ca/epidemiology/joseph/courses/epib-621/logfit.pdf AND https://support.sas.com/resources/papers/proceedings14/1485-2014.pdf 
def Hosmer_Lemeshow(res, df, bins=10): #for ungrouped data
    pred_raw = res.predict(df)
    pred= pred_raw.round()
    group = pd.qcut(x=pred_raw, q=bins, labels=False)
    
    new_df = pd.DataFrame({"pred_raw":pred_raw, "pred":pred, "group":group, "y":res.model.endog})

    H = 0
    for i,g in new_df.groupby("group"):
        exp_t = g["pred_raw"].sum()
        obs_t = (g["y"]==1).sum()

        exp_f = len(g)-exp_t
        obs_f = (g["y"]==0).sum()
        new = (obs_t - exp_t)**2 / exp_t + (obs_f - exp_f)**2 / exp_f
        H += new

    p = 1 - chi2.cdf(H, bins-2)
    return p, H

def standard_pearson(res, df): #for grouped data
    x = res.model.exog_names[1:]
    raw_pred = res.predict(df[x])

    def inner_fn(g):
        yhat_i = raw_pred[g.index].iloc[0] #more numerically stable
        y_i = g[y].mean()
        r = (y_i - yhat_i)**2 / (yhat_i*(1-yhat_i))
        return r

    test_stat = df.groupby(x).apply(lambda g:inner_fn(g)).sum()
    p = 1 - chi2.cdf(test_stat, res.df_resid)
    return p, test_stat

def bash_interaction(good_cols):
    q= []
    for a,b in itertools.combinations(good_cols, 2):
          q.append(f"{a}:{b}")
    return " + ".join(q)


#main function to run logreg
def run(y, df, reg=True, grouped=False):
    formula = y + " ~ " + " + ".join(x)
    mod = smf.mnlogit(formula=formula, data=df)
    if reg:
        res = mod.fit_regularized(method="l1", disp=0)
    else:
        res = mod.fit(method="bfgs", maxiter=1000, disp=0)
    significant_vars = res.pvalues[res.pvalues < 0.05].index
    if 'Intercept' in significant_vars:
        significant_vars = significant_vars.drop('Intercept')
    percents = [(name, np.exp(res.params[name])-1) for name in significant_vars]
    percents.sort(key = lambda x: abs(x[1]), reverse=True)
    plt.title(f"Torndo Plot for {y}")
    sns.barplot(x=[z[1] for z in percents], y=[z[0] for z in percents], orient='h')
    print("Psudo R2", res._results.prsquared)

    print("LINK", link_test(res))
    if grouped:
        print("PEARSON", standard_pearson(res, df))
    else:
        print("HOSMER", Hosmer_Lemeshow(res))

    return res


#run train/test split
def run_split(df, y_var):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    res = run(y_var, df_train, reg=False, grouped=False)
    yhat = res.predict(df_test)
    yhat = yhat.idxmax(axis=1)
    yhat = yhat + 1
    yhat = yhat.astype(int)
    y_test = df_test[y_var]
    print(classification_report(y_test, yhat))


res = run(y, df)