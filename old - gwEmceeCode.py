import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from GRWaveMaker import GRWaveMaker


def model(theta):
    #M1, dm, phic, tc = theta[0], theta[1], theta[2], theta[3]
    return gwe.hp4(times, theta)

def lnLike(theta):
    return -np.sum((model(theta)-signal)**2) / (2*var)  

def lnPrior(theta):
    M1, dM, phic, tc = theta[0], theta[1], theta[2], theta[3]
    if 0.0 < M1 < 40.0 and 0.0 < dM < M1 and 0 < phic < 2*np.pi and -200 < tc < 1000:
        return 0
    return -np.inf

def lnProb(theta):
    lp = lnPrior(theta)
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(lnLike(theta)):
        print "lnLike(theta) is infinite", theta
    return lp + lnLike(theta)

def makeSamplerGR4(nWalkers, timesteps):
    nDim = 4

    pos = [[M1_GR, deltaM, phic, tc] + 1e-2*np.random.randn(nDim) for i in range(nWalkers)] #the initial positions for the walkers

    sampler = emcee.EnsembleSampler(nWalkers, nDim, lnProb)
    sampler.reset()
    sampler.run_mcmc(pos, timesteps)
    #sampler.chain is a np array with size [nWalkers, timesteps, nDim]
    #sampler.flatchain is a np array with size [nWalkers*timesteps, nDim]

    return sampler

def makePlots(sampler, burnIn):
    samples = sampler.chain[:,burnIn:, :].reshape((-1,4))
    makeWalkerPlot(sampler)
    makeCornerPlot(samples)    

def makeCornerPlot(samples):
    print "making corner plot"
    fig = corner.corner(samples, labels=["$M_1$", "$\delta m$", "$\phi_c$", "$t_c$"], truths=[M1_GR, deltaM, phic, tc])
    fig.savefig("plots/triangle.png")

def makeWalkerPlot(sampler):
    print "making walker plot"
    plt.clf()
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].set_ylabel("$M_1$")
    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].set_ylabel("$\delta M$")
    axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
    axes[2].set_ylabel("$\phi_c$")
    axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
    axes[3].set_ylabel("$t_c$")

    fig.tight_layout(h_pad=0.0)
    fig.savefig("plots/walkers.png")

def makeSignal(signalStart, signalSamples, signalTimestep):
    times = -np.arange(signalSamples)*signalTimestep + signalStart
    #signal = np.random.normal(waveMaker.makeWave(GRProperties.m1, GRProperties.m2, GRProperties.phic, GRProperties.omega0, GRProperties.tc, times)) #The signal to be matched with uniform errors
    signal = waveMaker.makeWave(WWProperties.m1, WWProperties.m2, WWProperties.phic, WWProperties.omega0, WWProperties.tc, times)   
    return signal

sigmaAmp = 1.96         #result found from UniformErrors.py
var = sigmaAmp**2
M1_GR = 35.4			#Real Values of signal
M2_GR = 29.8                            
phic = np.pi
tc = 0
deltaM = M1_GR-M2_GR
M = M1_GR + M2_GR
waveMaker = GRWaveMaker()

class GRProperties:
    m1 = 35.4
    m2 = 29.8                            
    phic = 0
    omega0 = np.pi * 10
    tc = 0

class WWProperties:
    m1 = 1.4
    m2 = 10                            
    phic = 0
    omega0 = np.pi * 10
    tc = 0





