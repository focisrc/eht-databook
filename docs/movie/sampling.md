---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Sampling

We discuss an observation strategy for obtain a movie of M87*.

Scientifically, we are interested in the statistical properties of an M87* movie, instead of a specific movie realization.
In other word, we are interested in measuring the structure function of M87*.
For simplicity, we assume EHT is capable of obtaining an uniform resolution movie.
We focus on estimating the time structure function for each of the pixel in such movie.

## Definitions

Suppose we work with a discrete time series $f_i = f(t_i)$.
Let $\tau_k = t_j - t_i > 0$ be a lag, the (2nd-order) discrete structure function is defined by
$$
SF(\tau_k) = \sum_{i,j}\frac{[f_i - f_j]^2}{n_k},
$$
where $n_k$ is the number of different $i,j$ pairs.
Note that this definition does not assume uniform time sampling $t_i = i\Delta t$.

Nevertheless, if the function is periodic, uniform time sampling provide the same number of sampling at each lag.

+++

## Non-Periodic Function

However, this is not true for non-periodic functions.
In fact, the number of sampling at each lag decreases as the lag increase.
To see this, we can simply plot the histogram of the lags.

```{code-cell} ipython3
import numpy as np

def count(t):
    tau   = np.arange(len(t)+1)
    n_tau = np.zeros(len(t)+1)

    for ti in t:
        for tj in t:
            if tj > ti:
                tauk = tj - ti
                n_tau[round(tauk)] += 1
    return tau, n_tau
```

```{code-cell} ipython3
from matplotlib import pyplot as plt

N = 64
t = np.arange(N)

tau, n_tau = count(t)
plt.step(tau, n_tau, where='mid')
```

Similar trend appears for non-uniform sampling.

```{code-cell} ipython3
t = np.sort(np.random.uniform(size=N)*N)

tau, n_tau = count(t)
plt.step(tau, n_tau, where='mid')
```

```{code-cell} ipython3

```
