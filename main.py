
import numpy as np
np.random.seed(123)

import emcee
import matplotlib.pyplot as plt
import time
import os
import corner
from GRWaveMaker import GRWaveMaker

class GRProperties:
    m1 = 35.4
    m2 = 29.8                            
    phic = 0.0
    tc = 0.0
    signalStart = -0.1
    signalSamples = 1000
    signalTimestep = 0.01
    std = 1.0

class main():

    def __init__(self):
        cwd = os.getcwd()

        self.nDim            = 4 #M1, M2, phi_c, t_c
        self.nWalkers        = 50
        self.burnIn          = 50
        self.nSteps          = 100

        self.m1 = 35.4
        self.m2 = 29.8
        self.phic = 0
        self.omega0 = np.pi * 10
        self.tc = 0

        self.waveMaker = GRWaveMaker()

        self.figFolderName = "plots"
        if not os.path.exists(cwd + "/" + self.figFolderName):
            os.makedirs(cwd + "/" + self.figFolderName)

        self.dataFolderName = "data"
        if not os.path.exists(cwd + "/" + self.dataFolderName):
            os.makedirs(cwd + "/" + self.dataFolderName)

    def __setParams(self, Params):
        if Params == "WW":        
            self.m1 = WWProperties.m1
            self.m2 = WWProperties.m2
            self.phic = WWProperties.phic
            self.tc = WWProperties.tc
            self.signalStart     = WWProperties.signalStart
            self.signalSamples   = WWProperties.signalSamples
            self.signalTimestep  = WWProperties.signalTimestep    
            self.std = WWProperties.std
            self.var = self.std**2    
        else:
            if Params == "GR":        
                self.m1 = GRProperties.m1
                self.m2 = GRProperties.m2
                self.phic = GRProperties.phic
                self.tc = GRProperties.tc
                self.signalStart     = GRProperties.signalStart
                self.signalSamples   = GRProperties.signalSamples
                self.signalTimestep  = GRProperties.signalTimestep    
                self.std = GRProperties.std
                self.var = self.std**2    
            else:
                print "---Error: Params", Params, "not a valid option"
                return

    def __setGandC0(self, GandC0):
        if GandC0 == "natural":
            self.waveMaker.c0 = 1
            self.waveMaker.Gm_c03 = 1
        else:
            if GandC0 == "si":
                self.waveMaker.c0 = 299782.
                self.waveMaker.Gm_c03 = 4.925 * 10**(-6)
            else:
                print "---Error: GandC0", GandC0, "not a valid option"
                return

    def plotWaveandFrequencyRange(self, Params = "WW", GandC0 = "si"):
        
        self.__setParams(Params)
        self.__setGandC0(GandC0)

        times = -np.arange(self.signalSamples) * self.signalTimestep + self.signalStart
        signal = self.waveMaker.makeWave(self.m1, self.m2, self.phic, self.tc, times)   
        frequency = self.waveMaker.omega / (2 * np.pi)

        plt.figure(1)
        plt.subplot(211)
        plt.title("Inspiral for (10 M${}_\odot$, 1.4M${}_\odot$) non-spining system (WW fig 8)")
        plt.plot(times, frequency)
        plt.ylabel('Orbital Frequency')

        plt.subplot(212)
        plt.plot(times, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('$R h_+$')

        plt.savefig(self.figFolderName + "/" + Params + " " + GandC0 + " units signal signalStart %f signalSamples %d signalTimestep %f (WW fig 8).pdf" % (self.signalStart, self.signalSamples, self.signalTimestep), bbox_inches='tight')
        plt.cla()

        print "frequency hight", frequency[0], "frequency low", frequency[-1]

    def emcee(self, Params = "WW", GandC0 = "si"):

        self.__setParams(Params)
        self.__setGandC0(GandC0)

        self.times = -np.arange(self.signalSamples) * self.signalTimestep + self.signalStart
        self.signal = self.waveMaker.makeWave(self.m1, self.m2, self.phic, self.tc, self.times)
        self.signal = np.random.normal(self.signal, self.std)
        
        pos = [[self.m1, self.m2, self.phic, self.tc] + np.random.randn(self.nDim) for i in range(self.nWalkers)] #the initial positions for the walkers
        sampler = emcee.EnsembleSampler(self.nWalkers, self.nDim, self.__lnProb)
        sampler.run_mcmc(pos, self.nSteps)


        self.__makeWalkerPlot(sampler)
        #self.__makeCornerPlot(sampler.chain.reshape((-1,4)))

    def __lnProb(self, theta):
        lp = self.__lnPrior(*theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.__lnLike(*theta)

    def __lnLike(self, m1, m2, phic, tc):
        temp = - 1 / (2*self.var) * np.sum((self.waveMaker.makeWave(m1, m2, phic, tc, t = self.times + tc ) - self.signal)**2)
        return  temp

    def __lnPrior(self, m1, m2, phic, tc):
        if 0.0 < m1 < 40.0 and 0.0 < m2 < 40.0 and 0.0 < phic < 2*np.pi and -10.0 < tc < 10.0:
            return 0.0
        return -np.inf

    def __makeWalkerPlot(self, sampler):
        print "making walker plot"
        plt.clf()

        fig, axes = plt.subplots(self.nDim, figsize=(10, 7), sharex=True)
        samples = sampler.chain #samples = [walkers, steps, dim]

        labels = ["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s"]
        for i in range(self.nDim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");

        fig.tight_layout(h_pad=0.0)
        fig.savefig("plots/walkers.pdf", bbox_inches='tight')

    def __makeCornerPlot(self, samples):
        print "making corner plot"
        fig = corner.corner(samples, labels=["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s"], truths=[self.m1, self.m2, self.phic, self.tc])
        fig.savefig("plots/triangle.pdf")

class WWProperties:
    #Do Not Change!!
    m1 = 1.4
    m2 = 10.0                         
    phic = 0.0
    tc = 0.0
    signalStart = -0.0045
    signalSamples = 500
    signalTimestep = 0.00025
    std = 1.0


if __name__ == "__main__":
    start_time = time.time()
    myClass = main()
    #myClass.plotWaveandFrequencyRange("WW", "si")
    #myClass.plotWaveandFrequencyRange("WW", "natural")
    myClass.emcee()
    print("--- %s seconds ---" % (time.time() - start_time))


