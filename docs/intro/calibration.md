---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Calibration

```{code-cell} ipython3
import numpy as np
from math import pi
from matplotlib import pyplot as plt
```

## Add Delay Rate

$\Delta\phi(\nu,t) = \phi_0 + \frac{\delta\phi}{\delta\nu}\Delta\nu + \frac{\delta\phi}{\delta t}\Delta t$

Last term is delay rate.

$\Delta\phi_{12}(t) = \phi_{12,0} + (\frac{\partial\phi_1}{\partial t} - \frac{\partial\phi_2}{\partial t})\Delta t$

```{code-cell} ipython3
r  = lambda t: 1e-6 * t + 1
t  = np.linspace(0, 10_000, num=100_000)
s1 = np.sin(2 * pi * r(t) * t)
s2 = np.sin(2 * pi * r(t) * t + 0.123)
```

```{code-cell} ipython3
n1 = np.random.normal(scale=1, size=100_000)
n2 = np.random.normal(scale=1, size=100_000)
```

```{code-cell} ipython3
S1 = np.fft.fft(s1 + n1)
S2 = np.fft.fft(s2 + n2)
FX = S1 * np.conj(S2) # warning on convention
```

```{code-cell} ipython3
plt.semilogy(abs(FX[10_000-1000:10_000+1000]))
```

```{code-cell} ipython3
plt.plot(np.angle(FX[10_000-1000:10_000+1000]))
```

```{code-cell} ipython3
V = FX[10_000]

print(abs(V))
print(np.angle(V))
```

```{code-cell} ipython3
S1 = np.fft.fft((s1).reshape(100,1000),axis=1)
S2 = np.fft.fft((s2).reshape(100,1000),axis=1)
FX = S1 * np.conj(S2) # warning on convention
```

```{code-cell} ipython3
for i in range(100):
    plt.semilogy(abs(FX[i,:]))
```

```{code-cell} ipython3
for i in range(100):
    plt.plot(np.angle(FX[i,100-10:100+10]))
```

```{code-cell} ipython3

```
