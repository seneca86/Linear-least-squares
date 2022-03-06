# %%
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 150
sns.set_theme()
# %%
import pandas as pd
import numpy as np
df = pd.read_csv('.lesson/assets/FemPreg.csv')
# %%
def LeastSquares(xs, ys):
    mean_x = np.mean(xs)
    var_x = np.var(xs)
    mean_y = np.mean(ys)
    cov = np.dot(xs - mean_x, ys - mean_y) / len(xs)
    slope = cov / var_x
    inter = mean_y - slope * mean_x
    return inter, slope

df = df.dropna(subset=['agepreg', 'totalwgt_lb'])
inter, slope = LeastSquares(df.agepreg, df.totalwgt_lb)

# %%
def FitLine(xs, inter, slope):
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys

# %%
fit_xs, fit_ys = FitLine(df.agepreg, inter, slope)
# %%
df_fit = pd.DataFrame()
df_fit['agepreg'] = fit_xs
df_fit['totalwgt_lb'] = fit_ys
df_fit['type'] = 'fit'

df_train = df.loc[:,['agepreg', 'totalwgt_lb']]
df_train['type'] = 'train'
# %%
def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res
# %%
res = Residuals(df_train.agepreg, df_train.totalwgt_lb, inter, slope)
df_res = pd.DataFrame()
df_res['agepreg'] = fit_xs
df_res['totalwgt_lb'] = res
df_res['type'] = 'res'
# %%
df_ols = pd.concat([df_fit, df_train, df_res])
# %%
from pathlib import Path
Path('plots').mkdir(parents=True, exist_ok=True)
# %%
import seaborn as sns
import matplotlib.pyplot as plt
scatter1 = sns.relplot(
    data=df_ols,
    x="agepreg",
    y='totalwgt_lb',
    style='type',
    palette='deep',
    hue='type',
    markers={'train':'X', 'fit':'X', 'res':'X'},
    kind='scatter',
    col='type'
    )
scatter1.savefig('plots/scatter.png')
plt.clf()
# %%
def SampleRows(df, nrows, replace):
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample
# %%
def SamplingDistribution(df, iters=100):
    t = []
    for j in range(iters):
        sample = SampleRows(df, len(df), replace=True)
        estimates = LeastSquares(sample.agepreg, sample.totalwgt_lb)
        t.append(estimates)
    inters, slopes = zip(*t)
    return inters, slopes
# %%
inters, slopes = SamplingDistribution(df)
# %%
# %% Goodness of fit
def CoefDetermination(ys, res):
    return 1-(np.var(res)/np.var(ys))
# %%
CoefDetermination(df_ols.query('type=="train"').totalwgt_lb,
                  df_ols.query('type=="res"').totalwgt_lb)
# %% Testing a linear model
from dataclasses import dataclass
@dataclass
class SlopeTest:
    data: pd.DataFrame
    ages = data.iloc[:,0]
    weights = data.iloc[:,1]

    def TestStatistic(self):
        _, slope = LeastSquares(self.ages, self.weights)
        return slope
    
    def MakeModel(self):
        self.ybar = self.weights.mean()
        res = self.weights - self.ybar
        return res
    
    def RunModel(self):
        ages, _ = self.data
        weights = self.ybar + np.random.permutation(self.res)
        return ages, weights
# %%
live = df.dropna(subset=['agepreg', 'totalwgt_lb'])
# %%
ht = SlopeTest(live[['agepreg', 'totalwgt_lb']])
ht.TestStatistic()
ht.MakeModel()
# %%

# %%
