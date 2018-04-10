
import numpy as np
np.random.seed(123)

import emcee
import matplotlib.pyplot as plt
import time
import os
import corner
from GRWaveMaker import GRWaveMaker
from scipy import stats

class mcmcParams:
    nWalkers = 100
    burnIn = 500
    nSteps = 1500

class GRProperties:
    m1 = 35.4
    m2 = 29.8                            
    phic = 0.0
    tc = 0.0
    
    signalFq = 300.0

    signalTimestep = 1/signalFq
    signalStart = -signalTimestep
    signalSamples = int(0.2 / signalTimestep)

    std = 2.5 # needs to be a float
    #ligo samples at 16384 hz. online data downsamples to 4096 hz
    # "In their most sensative band, 100-300Hz, ..."
    #Over 0.2 s, the sifnal increases in frequency and amplitude in about 8 cycles from 35 hz to 150 hz

class WWProperties:
    # will recover WW fig 8
    m1 = 10
    m2 = 1.4                         
    phic = 0.0
    tc = 0.0
    signalStart = -0.0045
    signalSamples = 500
    signalTimestep = 0.00025
    std = 1


class main():

    def __init__(self, Params):
        cwd = os.getcwd()

        self.nWalkers        = mcmcParams.nWalkers
        self.burnIn          = mcmcParams.burnIn
        self.nSteps          = mcmcParams.nSteps

        self.m1 = 35.4
        self.m2 = 29.8
        self.phic = 0
        self.omega0 = np.pi * 10
        self.tc = 0

        self.ln2pi = np.log(2*np.pi)

        self.waveMaker = GRWaveMaker()

        self.Params = Params
        self.setParams(Params)
        self.__setGandC0("si")

        self.baseFigFolderName = "plots"
        if not os.path.exists(cwd + "/" + self.baseFigFolderName):
            os.makedirs(cwd + "/" + self.baseFigFolderName)

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
                print ("---Error: Params", Params, "not a valid option")
                raise ValueError, "Params not found"

        self.m1 = className.m1
        self.m2 = className.m2
        self.phic = className.phic
        self.tc = className.tc
        self.alpha = 0.5
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
                print ("---Error: GandC0", GandC0, "not a valid option")
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

        print ("frequency hight", frequency[0], "frequency low", frequency[-1], "\n")

    def emcee(self, mod = False, R = 1):
        
        cwd = os.getcwd()

        if mod == False:
            self.figFolderName = self.baseFigFolderName + "/gr_wave_mcmc"
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)
            self.figFolderName += "/params_" + self.Params + "_nWalkers_%d_nSteps_%d_burnIn_%d" % (self.nWalkers, self.nSteps, self.burnIn)
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)
        else:
            self.figFolderName = self.baseFigFolderName +  "/mod_wave_mcmc"
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)
            self.figFolderName += "/params_" + self.Params + "_nWalkers_%d_nSteps_%d_burnIn_%d_R_%d" % (self.nWalkers, self.nSteps, self.burnIn, R)
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)


        self.times = -np.arange(self.signalSamples) * self.signalTimestep + self.signalStart
        self.waveMaker.setMod(False)
        wave = self.waveMaker.makeWave(self.m1, self.m2, self.phic, self.tc, self.times)
        self.signal = np.random.normal(wave, self.std)


        self.waveMaker.setMod(mod)

        if mod == False:
            self.nDim = 4
            self.__lnPrior = self.__lnPriorGR
            pos = [[self.m1, self.m2, self.phic, self.tc] + np.random.randn(self.nDim) * 0.0001 for i in range(self.nWalkers)] #the initial positions for the walkers
        else:
            self.nDim = 5
            self.__lnPrior = self.__lnPriorMod
            self.waveMaker.R = R        
            pos = [[self.m1, self.m2, self.phic, self.tc, self.alpha] + np.random.randn(self.nDim) * 0.0001 for i in range(self.nWalkers)] #the initial positions for the walkers

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

            if mod == True:
                if params[4] < 0:
                    params[4] = 0
                if params[4] > 1:
                    params[4] = 1

        sampler = emcee.EnsembleSampler(self.nWalkers, self.nDim, self.__lnProb)
        print ("Starting mcmc")
        sampler.run_mcmc(pos, self.nSteps)
        

        #if we want to burn in 
        chain = sampler.chain[:, self.burnIn:, :].reshape((-1,self.nDim))
        #chain = sampler.flatchain

        #self.__makeWalkerPlot(sampler)
        self.__makeCornerPlot(chain.reshape((-1,self.nDim)))
        #self.__makeMeanTrueSignalPlot(self.times, chain, wave, self.signal)


    def __lnProb(self, theta):
        #m1, m2, phic, tc (,alpha) = theta
        lp = self.__lnPrior(*theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.__lnLike(theta)

    def __lnPriorMod(self, m1, m2, phic, tc, alpha):

        if 0 < m1 < 40 and 0 < m2 < m1 and 0 <= phic < 2*np.pi and 0 <= tc < 10.0 and 0 < alpha < 1:
            return 0.0
        return -np.inf

    def __lnPriorGR(self, m1, m2, phic, tc):

        if 0 < m1 < 40 and 0 < m2 < m1 and 0 <= phic < 2*np.pi and 0 <= tc < 10.0:
            return 0.0
        return -np.inf

    def __lnLike(self, theta):
        wave = self.waveMaker.makeWave(*theta, t = self.times)
        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * self.var ) - 0.5 *  self.invVar * np.sum( (wave - self.signal[(len(self.signal) - len(wave)):])**2 ) 
        return  temp

    def __makeWalkerPlot(self, sampler):
        print ("making walker plot")
        plt.clf()

        fig, axes = plt.subplots(self.nDim, figsize=(10, 7), sharex=True)
        plt.suptitle("Walkers from MCMC for (%f M${}_\odot$, %f M${}_\odot$) non-spining system" % (self.m1, self.m2))
        
        samples = sampler.chain #samples = [walkers, steps, dim]

        if self.waveMaker.mod == False:
            labels = ["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s"]
        else:
            labels = ["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s", "alpha"]


        for i in range(self.nDim):
            ax = axes[i]
            ax.plot(samples[:, :, i].T, "k", alpha=0.3)
            ax.set_xlim(0, len(samples[:, :, 0].T))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");

        #fig.tight_layout()
        fig.savefig(self.figFolderName + "/" + self.params + " " + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f walker.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')

    def __makeCornerPlot(self, samples):
        print ("making corner plot")

        if self.waveMaker.mod == False:
            fig = corner.corner(samples, labels=["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s"], truths=[self.m1, self.m2, self.phic, self.tc], show_titles = True)
        else:
            fig = corner.corner(samples, labels=["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ /s", "alpha"], truths=[self.m1, self.m2, self.phic, self.tc, 1], show_titles = True)

        fig.savefig(self.figFolderName + "/" + self.params + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f corner.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')

    def __makeMeanTrueSignalPlot(self, times, chain, wave, signal):
        print ("making mean true signal plot")
        plt.clf()
        plt.plot(times, signal, label="Signal")
        plt.plot(times, wave, label="True")
        print ("finding mean params")
        meanParams = [np.mean(param) for param in chain.T]
        print ("meanParams",  meanParams)
        plt.plot(times, self.waveMaker.makeWave(*meanParams, t = times), label="Emcee mean value")
        print ("finding mode params")

        print "np.percentile(x, 50)", np.percentile(chain.T, 50)

        #modeParams = [stats.mode(param) for param in chain.T]
        #modeParams = [param[0] for param in modeParams ]
        #print ("modeParams",  modeParams)

        #plt.plot(times, self.waveMaker.makeWave(*modeParams, t = times), label="Emcee mode value")
        plt.legend(loc = "best")
        plt.title("Waveform for (%f M${}_\odot$, %f M${}_\odot$) non-spining system" % (self.m1, self.m2))
        plt.xlabel('Time (s)')
        plt.ylabel('$R h_+$')
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




if __name__ == "__main__":
    start_time = time.time()
    myClass = main(Params = "GR") # Will us Gw150914 properties
    myClass.emcee() # will run GR mcmc
    #myClass.emcee() # will run GR mcmc
    #myClass.emcee(mod = True, R = 1000) # will run mod mcmc
    #myClass.emcee(mod = True, R = 100) # will run mod mcmc
    #myClass.emcee(mod = True, R = 10) # will run mod mcmc
    #myClass.emcee(mod = True, R = 1) # will run mod mcmc
    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n")






















