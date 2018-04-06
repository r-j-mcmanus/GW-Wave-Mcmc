
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
        self.burnIn          = 500
        self.nSteps          = 1000

        self.m1 = 35.4
        self.m2 = 29.8
        self.phic = 0
        self.omega0 = np.pi * 10
        self.tc = 0

        self.ln2pi = np.log(2*np.pi)

        self.waveMaker = GRWaveMaker()

        self.figFolderName = "plots"
        if not os.path.exists(cwd + "/" + self.figFolderName):
            os.makedirs(cwd + "/" + self.figFolderName)

        self.dataFolderName = "data"
        if not os.path.exists(cwd + "/" + self.dataFolderName):
            os.makedirs(cwd + "/" + self.dataFolderName)

    def __setParams(self, Params):
        if Params == "WW":
            className = WWProperties

        else:
            if Params == "GR":        
                className = GRProperties
            else:
                print "---Error: Params", Params, "not a valid option"
                return

        self.m1 = className.m1
        self.m2 = className.m2
        self.phic = className.phic
        self.tc = className.tc
        self.signalStart     = className.signalStart
        self.signalSamples   = className.signalSamples
        self.signalTimestep  = className.signalTimestep    
        self.std = className.std
        self.var = self.std**2   
        self.invVar = 1 / self.var 



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

        print "frequency hight", frequency[0], "frequency low", frequency[-1], "\n"

    def emcee(self, Params = "WW", GandC0 = "si"):

        self.__setParams(Params)
        self.__setGandC0(GandC0)

        self.times = -np.arange(self.signalSamples) * self.signalTimestep + self.signalStart
        wave = self.waveMaker.makeWave(self.m1, self.m2, self.phic, self.tc, self.times)
        self.signal = np.random.normal(wave, self.std)

        pos = [[self.m1, self.m2, self.phic, self.tc] + np.random.randn(self.nDim) for i in range(self.nWalkers)] #the initial positions for the walkers

        for params in pos:
            if params[0] < 0:
                params[0] = -params[0]

            if params[1] < 0:
                params[1] = -params[1]

            if params[1] > params[0]: #m2<m1
                params[1] = params[0]

            if params[2] < 0:
                params[2] = -params[2]

            if params[2] > 2 * np.pi:
                params[2] = 2 * pi

            if params[3] < 0:
                params[3] = -params[3]

            if params[3] > 10:
                params[3] = 10

        sampler = emcee.EnsembleSampler(self.nWalkers, self.nDim, self.__lnProb)
        sampler.run_mcmc(pos, self.nSteps)
        
        chain = sampler.flatchain
        print chain
        meanParams = [np.mean(chain.T[i]) for i in range(self.nDim)] 
        print meanParams

        self.__makeWalkerPlot(sampler)
        self.__makeCornerPlot(sampler.chain.reshape((-1,self.nDim)))
        self.__makeMeanTrueSignalPlot(meanParams, wave, self.signal)



    def __lnProb(self, theta):
        m1, m2, phic, tc = theta
        lp = self.__lnPrior(m1, m2, phic, tc)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.__lnLike(m1, m2, phic, tc)

    def __lnPrior(self, m1, m2, phic, tc):
        if 0 < m1 < 40 and 0 < m2 < m1 and 0 < phic < 2*np.pi and 0 <= tc < 10.0:
            return 0.0
        return -np.inf

    def __lnLike(self, m1, m2, phic, tc):
        wave = self.waveMaker.makeWave(m1, m2, phic, tc, self.times)
        
        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * self.var ) - 0.5 *  self.invVar * np.sum( (wave - self.signal[(len(self.signal) - len(wave)):])**2 ) 
        
        return  temp




    def __makeWalkerPlot(self, sampler):
        print "making walker plot"
        plt.clf()

        fig, axes = plt.subplots(self.nDim, figsize=(10, 7), sharex=True)
        samples = sampler.chain #samples = [walkers, steps, dim]

        labels = ["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s"]
        for i in range(self.nDim):
            ax = axes[i]
            ax.plot(samples[:, :, i].T, "k", alpha=0.3)
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

    def __makeMeanTrueSignalPlot(self, meanParams, wave, signal):
        print "making mean true signal plot"
        plt.clf()
        plt.plot(self.times, signal, label="Signal")
        plt.plot(self.times, wave, label="True")
        plt.plot(self.times, self.waveMaker.makeWave(*meanParams, t = self.times), label="Emcee best value")
        plt.legend(loc = "best")
        plt.savefig(self.figFolderName + "/"  + "MeanTrueSignalPlot.pdf", bbox_inches='tight')

    def makeTestWave(self, m1, m2, phic, tc, times):
        self.__setParams("WW")
        self.__setGandC0("si")

        myClass.waveMaker.makeWave(m1, m2, phic, tc, times )

class WWProperties:
    #Do Not Change!!
    m1 = 10
    m2 = 1.4                         
    phic = 0.0
    tc = 0.0
    signalStart = -0.0045
    signalSamples = 500
    signalTimestep = 0.00025
    std = 0.01


if __name__ == "__main__":
    try:
        print "\n"
        start_time = time.time()
        myClass = main()
        #myClass.plotWaveandFrequencyRange("WW", "si")
        #myClass.plotWaveandFrequencyRange("WW", "natural")
        myClass.emcee()

        #m1 = 1.08e+01
        #m2 = 1.98e-5
        #phic = 4.06e-01
        #tc = 1.87
        #sampleTimes = -np.arange(50) * WWProperties.signalTimestep + WWProperties.signalStart

        #print "\ntestwave", m1, m2, phic, tc
        #myClass.makeTestWave(m1, m2, phic, tc, sampleTimes)

        print("--- %s seconds ---" % (time.time() - start_time))
    except ValueError, message:
        print "\nValueError", message






















