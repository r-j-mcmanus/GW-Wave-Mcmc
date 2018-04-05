
import numpy as np
#import gwEmceeCode as code
import matplotlib.pyplot as plt
import time
import os
from GRWaveMaker import GRWaveMaker

class GRProperties:
    m1 = 35.4
    m2 = 29.8                            
    phic = 0
    omega0 = np.pi * 10
    tc = 0
    signalStart = -0.1
    signalSamples = 1000
    signalTimestep = 0.01





class main():

    def __init__(self):
        cwd = os.getcwd()

        self.nDim            = 4 #M1, M2, phi_c, t_c
        self.nWalkers        = 50
        self.timesteps       = 300
        self.burnIn          = 50


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

    def plotWaveandFrequencyRange(self, Params = "WW", GandC0 = "natural"):

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

        if Params == "WW":        
            m1 = WWProperties.m1
            m2 = WWProperties.m2
            phic = WWProperties.phic
            omega0 = WWProperties.omega0
            tc = WWProperties.tc
            signalStart     = WWProperties.signalStart
            signalSamples   = WWProperties.signalSamples
            signalTimestep  = WWProperties.signalTimestep        
        else:
            if Params == "GR":        
                m1 = GRProperties.m1
                m2 = GRProperties.m2
                phic = GRProperties.phic
                omega0 = GRProperties.omega0
                tc = GRProperties.tc
                signalStart     = GRProperties.signalStart
                signalSamples   = GRProperties.signalSamples
                signalTimestep  = GRProperties.signalTimestep  
            else:
                print "---Error: Params", Params, "not a valid option"
                return

        times = -np.arange(signalSamples) * signalTimestep + signalStart
        signal = self.waveMaker.makeWave(m1, m2, phic, omega0, tc, times)   
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

        plt.savefig(self.figFolderName + "/" + Params + " " + GandC0 + " units signal signalStart %f signalSamples %d signalTimestep %f (WW fig 8).pdf" % (signalStart, signalSamples, signalTimestep), bbox_inches='tight')
        plt.cla()

        print "frequency hight", frequency[0], "frequency low", frequency[-1]

    #sampler = code.makeSamplerGR4(nWalkers, timesteps)

    #samples = sampler.chain[:, burnIn:, :].reshape((-1, nDim))
    #stats = [[np.mean(samples[:,i]), np.std(samples[:,i])] for i in range(nDim)]

    #np.savetxt(samplesFilename, samples)
    #np.savetxt(statsFilename, stats)
    #np.savetxt(sampelrFilenameM1, sampler.chain[:, :, 0])
    #np.savetxt(sampelrFilenameDM, sampler.chain[:, :, 1])
    #np.savetxt(sampelrFilenamePhic, sampler.chain[:, :, 2])
    #np.savetxt(sampelrFilenameTc, sampler.chain[:, :, 3])

    #code.makePlots(sampler, burnIn)

class WWProperties:
    #Do Not Change!!
    m1 = 1.4
    m2 = 10                            
    phic = 0
    omega0 = np.pi * 10
    tc = 0
    signalStart = -0.0045
    signalSamples = 500
    signalTimestep = 0.00025


if __name__ == "__main__":
    start_time = time.time()
    myClass = main()
    myClass.plotWaveandFrequencyRange("WW", "si")
    #myClass.plotWaveandFrequencyRange("WW", "natural")
    print("--- %s seconds ---" % (time.time() - start_time))


