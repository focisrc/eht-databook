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

The correlation step takes signals from a pair of telescopes and
computes the Fourier transform of their cross correlations.
A significant property of such an operation is to remove the noise
from the visibility data.

Let $s$ be the sky signal from some radio source.
When the sky signal arrives at telescopes 1 and 2, the recorded
signals $s_1$ and $s_2$, although may have a time delay due to
geometric effects, are correlated.
Let $n_1$ and $n_2$ be noise at each of the telescopes, the cross
correlation is
\begin{align}
  C &= \langle (s_1 + n_1) (s_2 + n_2) \rangle \\
    &= \langle s_1 s_2 \rangle
     + \langle s_1 n_2 \rangle
     + \langle s_2 n_1 \rangle
     + \langle n_1 n_2 \rangle.
\end{align}
All the terms other than first term should vanishes.

By taking the Fourier transform of such a cross correlation, we obtain
also the spectral information of the signal.

In this chapter, we will create synthetic very long baseline
interferometry (VLBI) data and demostrate how cross-correlation can
remove the noise.
We will also show the XF and FX correlators are mathematically
identical, although the FX correlator is computationally more
efficient.

To get started, we first import the standard python packages:

```{code-cell} ipython3
import numpy as np
from math import pi
from matplotlib import pyplot as plt
```

## Signal Generation

```{code-cell} ipython3
t  = np.linspace(0, 10_000, num=100_000)
s1 = np.sin(2 * pi * t)
s2 = np.sin(2 * pi * t + 0.123)
```

```{code-cell} ipython3
plt.plot(t[:20], s1[:20])
plt.plot(t[:20], s2[:20])
```

## XF Correlator

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

## FX Correlator

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

## Introducing Noise

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
