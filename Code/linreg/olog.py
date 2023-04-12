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

df = pd.read_csv()
x = list(df.columns)

y = ""
x.remove(y)

# %%
formula = y + " ~ " + " + ".join(x)
mod = OrderedModel.from_formula(formula, data=df, distr="logit")
res = mod.fit(method='bfgs', disp=False)