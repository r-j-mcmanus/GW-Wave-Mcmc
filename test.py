

from GRWaveMaker import GRWaveMaker

import matplotlib.pyplot as plt

import numpy as np


signalFq = 300.0

signalTimestep = 1/signalFq
signalStart = -signalTimestep
signalSamples = int(0.2 / signalTimestep)

times = -np.arange(signalSamples) * signalTimestep + signalStart


m1 = 35.4
m2 = 29.8                            
phic = 0.0
tc = 0.0

waveMaker = GRWaveMaker()

waveMaker.setMod(False)
waveMaker.c0 = 299782.
waveMaker.Gm_c03 = 4.925 * 10**(-6)

GRwave = waveMaker.makeWave(m1, m2, phic, tc, times)
modBits = waveMaker.madeMod(m1, m2, phic, tc, 1, times)

plt.plot(times, GRwave)
plt.plot(times, modBits)
plt.show()
