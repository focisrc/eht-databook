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

# Correlation

```{code-cell} ipython3
import numpy as np
from math import pi
from matplotlib import pyplot as plt
```

## No Delay Rate

### Signal Generation

```{code-cell} ipython3
t  = np.linspace(0, 10_000, num=100_000)
s1 = np.sin(2 * pi * t)
s2 = np.sin(2 * pi * t + 0.123)
```

```{code-cell} ipython3
plt.plot(t[:20], s1[:20])
plt.plot(t[:20], s2[:20])
```

### XF Correlator

```{code-cell} ipython3
print(np.roll(np.arange(10), 1)) # <- this is convolution
print(np.roll(np.arange(10),-1)) # <- this is correlation
```

```{code-cell} ipython3
x = [np.mean(np.roll(s1,-tau) * s2) for tau in range(100_000)] # warning on convention
```

```{code-cell} ipython3
plt.plot(x[:20])
```

```{code-cell} ipython3
XF = np.fft.fft(x)
```

```{code-cell} ipython3
plt.semilogy(abs(XF))
```

```{code-cell} ipython3
n = np.argmax(abs(XF))

V = XF[n]

print(n)
print(abs(V))
print(np.angle(V))
```

### FX Correlator

```{code-cell} ipython3
S1 = np.fft.fft(s1)
S2 = np.fft.fft(s2)
FX = S1 * np.conj(S2) # warning on convention
```

```{code-cell} ipython3
plt.semilogy(abs(FX))
```

```{code-cell} ipython3
n = np.argmax(abs(FX))

V = FX[n]

print(n)
print(abs(V))
print(np.angle(V))
```

### Introducing Noise

```{code-cell} ipython3
n1 = np.random.normal(scale=1, size=100_000)
n2 = np.random.normal(scale=1, size=100_000)
```

```{code-cell} ipython3
plt.plot(s1 + n1)
plt.plot(s2 + n2)
```

```{code-cell} ipython3
S1 = np.fft.fft(s1 + n1)
S2 = np.fft.fft(s2 + n2)
FX = S1 * np.conj(S2) # warning on convention
```

```{code-cell} ipython3
plt.semilogy(abs(FX))
```

```{code-cell} ipython3
n = np.argmax(abs(FX))

V = FX[n]

print(n)
print(abs(V))
print(np.angle(V))
```
