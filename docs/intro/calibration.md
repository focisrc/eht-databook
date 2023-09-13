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

The visibility data obtained by correlation contains systematic
errors.
Many of them shows up as delays, which translate to phase errors in
visibility.
Source of delays include:

* Water vapor in troposphere
* Electron content in Ionosphere
* Instruments
* Clock inaccuracy
* ..
* Thermal noise

Fringe-fitting is a major calibration step to remove the delay and
delay rates in the visibility data so they can be coherencely averaged
to higher signal-to-noise data.

To get a sense on how this works, let's set up our numerical
experiment.
We again need to import the standard python packages.
We also turn fix the noise realizations and put our correlator in a
function.

```{code-cell} ipython3
import numpy as np
from math import pi, ceil
from matplotlib import pyplot as plt

t  = np.linspace(0, 10_000, num=100_000)
n1 = np.random.normal(scale=1, size=100_000)
n2 = np.random.normal(scale=1, size=100_000)

def chunked_FX(s1, s2, n=1000):
    N  = int(ceil(len(s1) / n))
    S1 = np.fft.rfft(np.pad(s1+n1, (0, N*n-len(s1))).reshape(N, n))
    S2 = np.fft.rfft(np.pad(s2+n2, (0, N*n-len(s2))).reshape(N, n))
    return np.conj(S1) * S2
```

## Adding Delay and Delay Rate

To visualize how the delay and delay rate affect the visibility, we
define $d(t)$ and $r(t)$ and apply them to the signal for the second
station.

```{code-cell} ipython3
d  = lambda t: 1e-3 * t/len(t) + 1
r  = lambda t: 1e-3 * t/len(t) + 1

s1 = np.sin(2 * pi * t)
s2 = np.sin(2 * pi * r(t) * (t + 0.123 / (2 * pi) * d(t)))

fig, (ax0, ax1, ax2) = plt.subplots(1,3, sharey=True, figsize=(12,4))
plt.subplots_adjust(wspace=0)

ax0.plot(t[:20], s1[:20])
ax0.plot(t[:20], s2[:20])
ax1.plot(t[50_000-10:50_000+10], s1[50_000-10:50_000+10])
ax1.plot(t[50_000-10:50_000+10], s2[50_000-10:50_000+10])
ax2.plot(t[-20:], s1[-20:])
ax2.plot(t[-20:], s2[-20:])

ax2.legend()
```

## Add Delay Rate

$\Delta\phi(\nu,t) = \phi_0 + \frac{\delta\phi}{\delta\nu}\Delta\nu + \frac{\delta\phi}{\delta t}\Delta t$

Last term is delay rate.

$\Delta\phi_{12}(t) = \phi_{12,0} + (\frac{\partial\phi_1}{\partial t} - \frac{\partial\phi_2}{\partial t})\Delta t$

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
