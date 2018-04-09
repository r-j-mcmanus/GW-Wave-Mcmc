
import numpy as np
np.random.seed(123)

import emcee
import matplotlib.pyplot as plt
import time
import os
import corner
from GRWaveMaker import GRWaveMaker
from scipy import stats


class GRProperties:
    m1 = 35.4
    m2 = 29.8                            
    phic = 0.0
    tc = 0.0
    
    signalFq = 1000.0

    signalTimestep = 1/signalFq
    signalStart = -signalTimestep
    signalSamples = int(0.2 / signalTimestep)

    std = 0.01
    #ligo samples at 16384 hz. online data downsamples to 4096 hz
    # "In their most sensative band, 100-300Hz, ..."
    #Over 0.2 s, the sifnal increases in frequency and amplitude in about 8 cycles from 35 hz to 150 hz

class main():

    def __init__(self, Params):
        cwd = os.getcwd()

        self.nDim            = 4 #M1, M2, phi_c, t_c
        self.nWalkers        = 100
        self.burnIn          = 100
        self.nSteps          = 1000

        self.m1 = 35.4
        self.m2 = 29.8
        self.phic = 0
        self.omega0 = np.pi * 10
        self.tc = 0

        self.ln2pi = np.log(2*np.pi)

        self.waveMaker = GRWaveMaker()

        self.setParams(Params)
        self.__setGandC0("si")

        self.figFolderName = "plots"
        if not os.path.exists(cwd + "/" + self.figFolderName):
            os.makedirs(cwd + "/" + self.figFolderName)
        self.figFolderName = "plots/" + Params + "_nWalkers_%d_nSteps_%d" % (self.nWalkers, self.nSteps)
        if not os.path.exists(cwd + "/" + self.figFolderName):
            os.makedirs(cwd + "/" + self.figFolderName)

        self.dataFolderName = "data"
        if not os.path.exists(cwd + "/" + self.dataFolderName):
            os.makedirs(cwd + "/" + self.dataFolderName)

    

    def setParams(self, Params):
        if Params == "WW":
            className = WWProperties
            self.params = Params
        else:
            if Params == "GR":        
                className = GRProperties
                self.params = Params
            else:
                print "---Error: Params", Params, "not a valid option"
                raise ValueError, "Params not found"

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
            self.units = GandC0
        else:
            if GandC0 == "si":
                self.waveMaker.c0 = 299782.
                self.waveMaker.Gm_c03 = 4.925 * 10**(-6)
                self.units = GandC0
            else:
                print "---Error: GandC0", GandC0, "not a valid option"
                raise ValueError, "Params not found"

    def plotWaveandFrequencyRange(self):

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

    def emcee(self):

        self.times = -np.arange(self.signalSamples) * self.signalTimestep + self.signalStart
        wave = self.waveMaker.makeWave(self.m1, self.m2, self.phic, self.tc, self.times)
        self.signal = np.random.normal(wave, self.std)

        pos = [[self.m1, self.m2, self.phic, self.tc] + np.random.randn(self.nDim) * 0.0001 for i in range(self.nWalkers)] #the initial positions for the walkers

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
        

        #if we want to butn in 
        chain = sampler.chain[:, self.burnIn:, :].reshape((-1,self.nDim))
        #chain = sampler.flatchain

        self.__makeWalkerPlot(sampler)
        self.__makeCornerPlot(chain.reshape((-1,self.nDim)))
        self.__makeMeanTrueSignalPlot(self.times, chain, wave, self.signal)


    def __lnProb(self, theta):
        m1, m2, phic, tc = theta
        lp = self.__lnPrior(m1, m2, phic, tc)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.__lnLike(m1, m2, phic, tc)

    def __lnPrior(self, m1, m2, phic, tc):
        if 0 < m1 < 40 and 0 < m2 < m1 and 0 <= phic < 2*np.pi and 0 <= tc < 10.0:
            return 0.0
        return -np.inf

    def _testLnPrior(self):
        print "m1 = 10, m2 = 10, phic = np.pi, tc = 2"
        print "__lnPrior", __lnPrior(10,10,np.pi, 2)

        print "m1 = -10, m2 = 10, phic = np.pi, tc = 2"
        print "__lnPrior", __lnPrior(-10,10,np.pi, 2)

        print "m1 = 10, m2 = -10, phic = np.pi, tc = 2"
        print "__lnPrior", __lnPrior(10,-10,np.pi, 2)

        print "m1 = 10, m2 = 10, phic = -np.pi, tc = 2"
        print "__lnPrior", __lnPrior(10,10,-np.pi, 2)

        print "m1 = 10, m2 = 10, phic = np.pi, tc = -2"
        print "__lnPrior", __lnPrior(10,10,np.pi, -2)

    def __lnLike(self, m1, m2, phic, tc):
        wave = self.waveMaker.makeWave(m1, m2, phic, tc, self.times)
        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * self.var ) - 0.5 *  self.invVar * np.sum( (wave - self.signal[(len(self.signal) - len(wave)):])**2 ) 
        return  temp

    def _testLnLike(self):

        times = -np.arange(WWProperties.signalSamples) * WWProperties.signalTimestep + WWProperties.signalStart
        wave = self._makeTestWave(times)
        signal = np.random.normal(wave, WWProperties.std)
        print "liklyhood for wave and wave" 
        #temp = - 0.5 * np.sum( (wave - wave[(len(wave) - len(wave)):])**2 ) / WWProperties.std**2
        #print "- 0.5 * np.sum( (wave - wave[(len(wave) - len(wave)):])**2 ) / WWProperties.std**2", temp
        #temp = - 0.5 * len(wave) * np.log( 2 * np.pi * WWProperties.std**2 )
        #print "- 0.5 * len(wave) * np.log( 2 * np.pi * WWProperties.std**2 )", temp
        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * WWProperties.std**2 ) - 0.5 * np.sum( (wave - wave[(len(wave) - len(wave)):])**2 ) / WWProperties.std**2
        print "liklyhood\n", temp
        print "liklyhood for wave and signal with std",  WWProperties.std
        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * WWProperties.std**2 ) - 0.5 * np.sum( (wave - signal[(len(signal) - len(wave)):])**2 ) / WWProperties.std**2
        print temp

        print "liklyhood for test wave and signal with std",  WWProperties.std
        wave = self._makeTestWave(times, 9.827, 0.389, 0.986, 0.230)
        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * WWProperties.std**2 ) - 0.5 * np.sum( (wave - signal[(len(signal) - len(wave)):])**2 ) / WWProperties.std**2
        print temp


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
        fig.savefig(self.figFolderName + "/" + self.params + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f walker.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')

    def __makeCornerPlot(self, samples):
        print "making corner plot"
        fig = corner.corner(samples, labels=["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s"], truths=[self.m1, self.m2, self.phic, self.tc], verbose = True)
        fig.savefig(self.figFolderName + "/" + self.params + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f corner.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')

    def __makeMeanTrueSignalPlot(self, times, chain, wave, signal):
        print "making mean true signal plot"
        plt.clf()
        plt.plot(times, signal, label="Signal")
        plt.plot(times, wave, label="True")
        print "finding mean params"
        meanParams = [np.mean(chain.T[i]) for i in range(self.nDim)]
        print "\nmeanParams",  meanParams, "\n"
        plt.plot(times, self.waveMaker.makeWave(*meanParams, t = times), label="Emcee mean value")
        print "finding mode params"
        modeParams = [stats.mode(chain.T[i]) for i in range(self.nDim)]
        modeParams = [modeParams[0][0],modeParams[1][0],modeParams[2][0],modeParams[3][0]]
        print "\nmodeParams",  modeParams, "\n"
        plt.plot(times, self.waveMaker.makeWave(*modeParams, t = times), label="Emcee mode value")
        plt.legend(loc = "best")
        plt.savefig(self.figFolderName + "/" + self.params + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f MeanModeTrueSignalPlot.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')   

    def _makeTestWave(self, times, m1 = 10, m2 = 1.4, phic = 0.0, tc = 0.0):
        return myClass.waveMaker.makeWave(m1, m2, phic, tc, times )

    def _makeOnlyMod(self, times, m1 = 10, m2 = 1.4, phic = 0.0, tc = 0.0, R = 1):

        signal = myClass.waveMaker.makeOnlyMod(m1, m2, phic, tc, times, R )

        plt.clf()
        plt.plot(times, signal)
        plt.title("Boundary contribution to inspiral for (%f M${}_\odot$, %f M${}_\odot$, R %f) non-spining system" % (m1, m2, R))
        plt.xlabel('Time (s)')
        plt.ylabel('$R h_+$')
        
        plt.savefig(self.figFolderName + "/" + self.params + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f Mod Wave.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep))   

class WWProperties:
    #Do Not Change!!
    m1 = 10
    m2 = 1.4                         
    phic = 0.0
    tc = 0.0
    signalStart = -0.0045
    signalSamples = 500
    signalTimestep = 0.00025
    std = 0.1


if __name__ == "__main__":
    try:
        start_time = time.time()
        myClass = main("GR")
        myClass.emcee()

        #m1 = 1.08e+01
        #m2 = 1.98e-5
        #phic = 4.06e-01
        #tc = 1.87
        #sampleTimes = -np.arange(50) * WWProperties.signalTimestep + WWProperties.signalStart

        #print "\ntestwave", m1, m2, phic, tc
        #myClass._makeTestWave(m1, m2, phic, tc, sampleTimes)

        #myClass._testLnLike()

        #myClass._makeOnlyMod(sampleTimes)

        print("--- %s seconds ---" % (time.time() - start_time))
        print "\n"
    except ValueError, message:
        print "\nValueError", message






















