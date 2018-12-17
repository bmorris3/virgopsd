import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.signal import periodogram

from virgopsd import generate_fluxes, kernel

# Generate 1e7 fluxes at 60 second cadence: 
# y = generate_fluxes(size=1e7, cadence=60*u.s)
y = generate_fluxes(size=1e7, cadence=1*u.s)
x = np.arange(len(y))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(x, y)
ax[0].set(xlabel='Time [min]', ylabel='Flux')

# ftest, Ptest = periodogram(y, fs=1/60)
ftest, Ptest = periodogram(y, fs=1)

ax[1].loglog(ftest * 1e6, Ptest, ',', label='Sample power')
ax[1].loglog(ftest * 1e6, 2*np.pi*kernel.get_psd(2*np.pi*ftest), alpha=0.7, label='Kernel')
ax[1].set_ylim([1e-10, 1e0])
ax[1].set_xlim([1e-2, 1e4])
ax[1].set(xlabel='Frequency [$\mu$Hz]', ylabel='Power')
ax[1].legend()

plt.show()