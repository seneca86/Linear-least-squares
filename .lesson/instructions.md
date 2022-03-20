## Linear least squares

# Least squares fit

There are several ways to estimate the slope of the relationship between two variables; the most common is a linear least squares fit. A __linear fit__ is a line intended to model the relationship between variables. A __least squares__ fit is one that minimizes the mean squared error (MSE) between the line and the data.

Suppose we have a sequence of points, `ys`, that we want to express as a function of another sequence `xs`. If there is a linear relationship between xs and ys with intercept inter and slope slope, we expect:

```python
y[i] = inter + slope * x[i]
```

But unless the correlation is perfect, there is a deviation from the line, or residual:

```python
res = ys - (inter + slope * xs)
```

We might try to minimize the absolute value of the residuals, or their squares, or their cubes; but the most common choice is to minimize the sum of squared residuals, `sum(res**2)`, because:

* Squaring treats positive and negative residuals the same

* Squaring gives more weight to large residuals

* If the residuals are uncorrelated and normally distributed with mean 0 and constant (but unknown) variance, then the least squares fit is also the maximum likelihood estimator of inter and slope. See [link](https://en.wikipedia.org/wiki/Linear_regression)

* The values of inter and slope that minimize the squared residuals can be computed efficiently

If you are using xs to predict values of ys, guessing too high might be better (or worse) than guessing too low. In that case you might want to compute some cost function for each residual, and minimize total cost, sum(cost(res)). However, computing a least squares fit is quick, easy and often good enough.

## Implementation

First, we will load a few libraries that we will need for our calculations and plotting:

```python
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi']= 150
sns.set_theme()
```

Then we will load the dataset:

```python
import pandas as pd
import numpy as np
df = pd.read_csv('.lesson/assets/FemPreg.csv')
```

And now we can define our own `LeastSquares` function:

```python
def LeastSquares(xs, ys):
    mean_x = np.mean(xs)
    var_x = np.var(xs)
    mean_y = np.mean(ys)
    cov = np.dot(xs - mean_x, ys - mean_y) / len(xs)
    slope = cov / var_x
    inter = mean_y - slope * mean_x
    return inter, slope
```

`LeastSquares` takes sequences `xs` and `ys` and returns the estimated parameters inter and slope. For details on how it works, see [here](http://wikipedia.org/wiki/Numerical_methods_for_linear_least_squares).

We can already apply the function to our dataset, if we first remove the `NA`s.

```python
df = df.dropna(subset=['agepreg', 'totalwgt_lb'])
inter, slope = leastsquares(df.agepreg, df.totalwgt_lb)
```

And now that we have the parameters of the model, we define a function to predict:

```python
def FitLine(xs, inter, slope):
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys
```

and apply it:

```python
fit_xs, fit_ys = FitLine(df.agepreg, inter, slope)
```

The estimated intercept and slope are 6.8 lbs and 0.017 lbs per year. These values are hard to interpret in this form: the intercept is the expected weight of a baby whose mother is 0 years old, which doesn’t make sense in context, and the slope is too small to grasp easily. Let's organize everything in a single dataframe:

```python
df_fit = pd.DataFrame()
df_fit['agepreg'] = fit_xs
df_fit['totalwgt_lb'] = fit_ys
df_fit['type'] = 'fit'

df_train = df.loc[:,['agepreg', 'totalwgt_lb']]
df_train['type'] = 'train'
```

### Residuals

An important check after performing a linear least squares fit is to calculate the residuals. Residuals takes sequences `xs` and `ys` and estimated parameters `inter` and `slope`. It returns the differences between the actual values and the fitted. 

Let's define a function for calculating the residuals:

```python
def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res
```

and apply it to our data:

```python
res = Residuals(df_train.agepreg, df_train.totalwgt_lb, inter, slope)
df_res = pd.DataFrame()
df_res['agepreg'] = fit_xs
df_res['totalwgt_lb'] = res
df_res['type'] = 'res'
```

As mentioned, it is useful to organize everything in the same dataframe:

```python
df_ols = pd.concat([df_fit, df_train, df_res])
```

For plotting purposes, let's load a few auxiliary functions and created a directory:

```python
from pathlib import Path
Path('plots').mkdir(parents=True, exist_ok=True)
import seaborn as sns
import matplotlib.pyplot as plt
```

Now we are ready to plot a scatterplot:

```python
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
```