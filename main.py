
import numpy as np
np.random.seed(123)

import emcee
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import corner
import cornerGR
import cornerMod
from GRWaveMaker import GRWaveMaker
from scipy import stats

counter = 0
lnlikeArray = []

class mcmcParams:
    nWalkers = 100
    burnIn = 1000
    nSteps = 2000
    #nWalkers = 20
    #burnIn = 50
    #nSteps = 200

class GRProperties:
    m1 = 35.4
    m2 = 29.8                            
    phic = 0.0
    tc = 0.0
    
    signalFq = 300.0

    signalTimestep = 1/signalFq
    signalStart = -signalTimestep
    signalSamples = int(0.2 / signalTimestep)

    std = 2.0   # needs to be a float
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
    std = 1.0


class main():

    def __init__(self, nWalkers = None, burnIn=None, nSteps=None, Params = "GR"):
        cwd = os.getcwd()

        if nWalkers == None:
            self.nWalkers = mcmcParams.nWalkers
        else: 
            self.nWalkers = nWalkers

        if burnIn == None:
            self.burnIn = mcmcParams.burnIn
        else: 
            self.burnIn = burnIn

        if nSteps == None:
            self.nSteps = mcmcParams.nSteps
        else: 
            self.nSteps = nSteps

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
        """
        Will set the paramiters for the binary system from the two classes GRProperties or WWProperties.

        Parameters
        ----------
        Params : string
            Pass either "WW" for WWProperties or "GR" for GRProperties
        """
        if Params == "WW":
            self.Properties = WWProperties
            self.params = Params
        else:
            if Params == "GR":        
                self.Properties = GRProperties
                self.params = Params
            else:
                print ("---Error: Params", Params, "not a valid option")
                raise ValueError, "Params not found"

        self.m1 = self.Properties.m1
        self.m2 = self.Properties.m2
        self.phic = self.Properties.phic
        self.tc = self.Properties.tc
        self.alpha = 0.5
        self.signalStart     = self.Properties.signalStart
        self.signalSamples   = self.Properties.signalSamples
        self.signalTimestep  = self.Properties.signalTimestep    
        self.std = float(self.Properties.std)
        self.var = self.std**2   
        self.invVar = 1 / self.var 



    def __setGandC0(self, GandC0 = "si"):
        """
        Will set the units for the constants G and c0.
        I would recomend no changing this.
        The code curently runs in si

        Parameters
        ----------
        GandC0 : string
            Pass either "si" for si units or "natural" for natural units
        """
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

    def __makeFigFolders(self, mod, R = 1.0):
        """
        Will make the folders to store any data and plots in.

        Parameters
        ----------
        mod : Bool
            Whether or not we are perfomring mcmc with a modified wave.

        R : float
            The size of the screening radius expressed in the specified units, currently set to si.
        """

        cwd = os.getcwd()

        if mod == False:
            self.figFolderName = self.baseFigFolderName + "/gr_wave_mcmc"
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)
            self.figFolderName += "/params_" + self.Params + "_nWalkers_%d_nSteps_%d_burnIn_%d" % (mcmcParams.nWalkers, mcmcParams.nSteps, mcmcParams.burnIn)
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)
        else:
            self.figFolderName = self.baseFigFolderName +  "/mod_wave_mcmc"
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)
            self.figFolderName += "/params_" + self.Params + "_nWalkers_%d_nSteps_%d_burnIn_%d_R_%d" % (mcmcParams.nWalkers, mcmcParams.nSteps, mcmcParams.burnIn, R)
            if not os.path.exists(cwd + "/" + self.figFolderName):
                os.makedirs(cwd + "/" + self.figFolderName)

    def emcee(self, mod = False, R = 1.0):
        """
        Will call emcee to find the best fit values for a gravitational wave from the signal made from 
        the passed paramiters on initialisation of the class.

        Will produce a corner plot of the mcmc output.

        Parameters
        ----------

        mod : bool
            whether or not we are comparing to a modified wave.

        R : float
            the size of the 
        """

        self.__makeFigFolders(mod, R)

        self.times = -np.arange(self.signalSamples) * self.signalTimestep + self.signalStart
        self.waveMaker.setMod(False)
        self.wave = self.waveMaker.makeWave(self.m1, self.m2, self.phic, self.tc, self.times)
        self.signal = np.random.normal(self.wave, self.std)


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
            params[0] = -params[0] if params[0] < 0 else params[0]

            params[1] = -params[1] if params[1] < 0 else params[1]
            #m2<m1
            params[1] = params[0] if params[1] > params[0] else params[1]

            params[2] = 0 if params[2] < 0  else params[2]
            params[2] = 2*np.pi if params[2] >  2*np.pi else params[2]

            #params[2] = -np.pi if params[2] < -np.pi  else params[2]
            #params[2] = np.pi if params[2] >  np.pi else params[2]

            params[3] = -params[3] if params[3] < 0 else params[3]
            params[3] = 10 if params[3] > 10 else params[3]

            if mod == True:
                params[4] = 0 if params[4] < 0 else params[4]
                params[4] = 1 if params[4] > 1 else params[4]

        sampler = emcee.EnsembleSampler(self.nWalkers, self.nDim, self.__lnProb)
        print "Starting mcmc"
        sampler.run_mcmc(pos, self.nSteps)
        
        chain = sampler.chain[:, self.burnIn:, :].reshape((-1,self.nDim))
        flatchain = chain.reshape((-1,self.nDim))

        #self.__makeWalkerPlot(sampler)
        self.__makeCornerPlot(flatchain)

        #self.__makeSignalPlot(flatchain)




    """-------------Fns for mcmc stuff----------------"""

    def __lnProb(self, theta):
        #m1, m2, phic, tc (,alpha) = theta
        lp = self.__lnPrior(*theta)
        if not np.isfinite(lp):
            return -np.inf
        lnlike = self.__lnLike(theta)

        if np.isnan(lnlike):
            print "\n----------Error: lnlike is nan------------"
            print "lnlike", lnlike, 
            print "lp", lp, "theta",  theta
            print "wave\n", self.waveMaker.makeWave(*theta, t = self.times)
            raise ValueError("lnlike is nan")

        return lp + lnlike

    def __lnPriorMod(self, m1, m2, phic, tc, alpha):
        #if 0 < m1 < 100 and 0 < m2 < m1 and -np.pi <= phic <= np.pi and -10 <= tc < 10.0 and 0 <= alpha <= 1:
        if 0 < m1 < 100 and 0 < m2 < m1 and 0 <= phic <= 2*np.pi and -10 <= tc < 10.0 and 0 <= alpha <= 1:
            return 0.0
        return -np.inf

    def __lnPriorGR(self, m1, m2, phic, tc):
        #if 0 < m1 < 100 and 0 < m2 < m1 and -np.pi <= phic <= np.pi and -10 < tc < 10.0:
        if 0 < m1 < 100 and 0 < m2 < m1 and 0 <= phic <= 2*np.pi and -10 < tc < 10.0:
            return 0.0
        return -np.inf

    def __lnLike(self, theta):
        wave = self.waveMaker.makeWave(*theta, t = self.times)
        wave2 = wave
        nanEnd = len(wave)
        for i in range(len(wave)):
            if not np.isnan(wave[i]):
                nanEnd = i
                break
        wave = wave[nanEnd:]

        temp = - 0.5 * len(wave) * np.log( 2 * np.pi * self.var ) - 0.5 *  self.invVar * np.sum( (wave - self.signal[(len(self.signal) - len(wave)):])**2 ) 

        if np.isnan(temp):
            print "-----error: lnLike returning a nan------"
            print "temp", temp
            print "nanEnd\n", nanEnd
            print "wave\n", wave
            print "wave2\n", wave2
            print "self.signal[(len(self.signal) - len(wave)):]\n", self.signal[(len(self.signal) - len(wave)):]
            print "- 0.5 * len(wave) * np.log( 2 * np.pi * self.var )",- 0.5 * len(wave) * np.log( 2 * np.pi * self.var )   
            print "- 0.5 *  self.invVar * np.sum( (wave - self.signal[(len(self.signal) - len(wave)):])**2 )", - 0.5 *  self.invVar * np.sum( (wave - self.signal[(len(self.signal) - len(wave)):])**2 )         
            raise ValueError

        return  temp



    """------------Fns for ploting stuff-------------"""

    def __makeWalkerPlot(self, sampler):
        """
        Makes a plot of the path the walkers traverse through the emcee run.

        Parameters
        ----------

        sampler : emcee.EnsembleSampler
            Pass the sampler after sampler.run_mcmc
        """

        print "making walker plot"
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

        fig.savefig(self.figFolderName + "/" + self.params + " " + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f walker.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')

    def __makeCornerPlot(self, flatchain):
        """
        Makes a corner plot from corner.py


        Parameters
        ----------
        flatchain : array[nsamples, ndim]
            Each row contains the values a parameter takes from an emcee run.
        """

        print ("making corner plot")

        K = self.nDim
        factor = 2.0           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.05         # w/hspace size
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim

        #make fig the right size
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))

        if self.waveMaker.mod == False:
            cornerGR.corner(flatchain, labels=["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ / s"], truths=[self.m1, self.m2, self.phic, self.tc], show_titles = True, fig = fig)
            #https://matplotlib.org/users/gridspec.html
            gs = gridspec.GridSpec(20, 20)
            ax = fig.add_subplot(gs[0:4, -9:])        

        else:
            cornerMod.corner(flatchain, labels=["M${}_1$ / M${}_\odot$", "M${}_2$ / M${}_\odot$", "$\phi_c$", "$t_c$ / s", r"$\alpha$"], show_titles = True, fig = fig)
            #https://matplotlib.org/users/gridspec.html 
            gs = gridspec.GridSpec(20, 20)
            ax = fig.add_subplot(gs[0:4, -9:])


        

        self.__makeSignalPlot(flatchain, axis = ax)

        fig.savefig(self.figFolderName + "/" + "params " + self.params + " " + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %.1f corner.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')
        

    def __makeSignalPlot(self, flatchain, axis = plt):
        """ 
        Will polt the signal and the best fit wave. if no argument for axis is made, will also save the plot.

        Parameters
        ----------
            flatchain : array[nsamples, ndim]
                Each row contains the values a parameter takes from an emcee run.

            axis : matplotlib.axes 
                If an instance is passed, the graph will be placed on the instance, 
                otherwise it will make a new plot and save it.
        """

        axis.plot(self.times, self.signal, 'b', label="Signal")
        if self.waveMaker.mod == False:
            axis.plot(self.times, self.wave, 'g', label="GR wave")
            plt.gcf().text(0.5, 0.95, "(a)", fontsize=18)
        else:
            plt.gcf().text(0.5, 0.95, "(a)", fontsize=22)

        #meanParams = [np.mean(param) for param in flatchain.T]
        #axis.plot(self.times, self.waveMaker.makeWave(*meanParams, t = self.times), label="Mean MCMC value")

        mostLikelyValues = self.__findMostLikely(flatchain)
        medianParams = [mostLikelyValue[0] for mostLikelyValue in mostLikelyValues]
        axis.plot(self.times, self.waveMaker.makeWave(*medianParams, t = self.times), 'r', label="Median MCMC value")

        axis.set_xlabel('Time (s)')
        axis.set_ylabel('$R h_+$')

        if axis == plt:
            axis.legend(loc = "best")
            axis.set_title("Waveform for (%.1f M${}_\odot$, %.1f M${}_\odot$) non-spining system" % (self.m1, self.m2))
            plt.savefig(self.figFolderName + "/" + self.params + self.units + " units signal signalStart %f signalSamples %d signalTimestep %f std %f MeanModeTrueSignalPlot.pdf" % (self.signalStart, self.signalSamples, self.signalTimestep, self.std), bbox_inches='tight')   
            return
    
    def __findMostLikely(self, flatchain):
        """
        Finds the most likly values for all paramiters and returns the list containing 
        tuples of 50 percentile, pluss minus for 90% and plus minus for 68%.

        Parameters
        ----------
            flatchain : array[nsamples, ndim]
                Each row contains the values a parameter takes from an emcee run.
        """
        mostLikelyValues = []
        for paramChain in flatchain.T:
            q_05, q_16, q_50, q_84, q_95 = np.percentile(paramChain, np.array([0.05, 0.16, 0.5, 0.84, 0.95])*100)
            q_m_90, q_p_90 = q_50-q_05, q_95-q_50
            q_m_68, q_p_68 = q_50-q_16, q_84-q_50
            mostLikelyValues.append((q_50, q_p_90, q_m_90, q_p_68, q_m_68))
        return mostLikelyValues



    def plotWaveandFrequencyRange(self):
        """
        Reproduces figure 8 from WW.
        """


        times = -np.arange(WWProperties.signalSamples) * WWProperties.signalTimestep + WWProperties.signalStart
        signal = self.waveMaker.makeWave(WWProperties.m1, WWProperties.m2, WWProperties.phic, WWProperties.tc, times)   
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

        plt.savefig(self.figFolderName + "/" + Params + " " + GandC0 + " units signal signalStart %f signalSamples %d signalTimestep %f (WW fig 8).pdf" % (WWProperties.signalStart, WWProperties.signalSamples, WWProperties.signalTimestep), bbox_inches='tight')
        plt.cla()

        print ("frequency hight", frequency[0], "frequency low", frequency[-1], "\n")



if __name__ == "__main__":
    start_time = time.time()
    myClass = main(Params = "GR") # Will us Gw150914 properties
    myClass.emcee(mod = False) # will run GR mcmc
    myClass.emcee(mod = True, R = 1) # will run mod mcmc
    #myClass.emcee(mod = True, R = 10) # will run mod mcmc
    #myClass.emcee(mod = True, R = 100) # will run mod mcmc
    #myClass.emcee(mod = True, R = 1000) # will run mod mcmc
    #myClass.emcee(mod = True, R = 10000) # will run mod mcmc
    #myClass.emcee(mod = True, R = 100000) # will run mod mcmc
    #floating point erros plague the values below this
    #myClass.emcee(mod = True, R = 1000000) # will run mod mcmc
    #myClass.emcee(mod = True, R = 10000000) # will run mod mcmc
    #myClass.emcee(mod = True, R = 100000000) # will run mod mcmc
    #myClass.emcee(mod = True, R = 1000000000) # will run mod mcmc
    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n")






















