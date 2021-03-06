
from numpy import cos, sin , log, pi, arange, isnan, isfinite, tan
import time
import sys


i = pi / 2.
c = cos(i) 
c2 = c**2
c4 = c**4
c6 = c**6
s = sin(i)
s2 = s**2
s4 = s**4
ln2 = log(2)
ln3_2 = log(3/2.)

counter = 1

class GRWaveMaker:
    
    def __init__(self):
        self.m1 = None
        self.m2 = None
        self.m = None
        self.dm = None
        self.mu = None
        self.eta = None
        self.eta2 = None
        self.Theta = None
        self.phi = None
        self.omega = None
        self.psi = None
        self.x = None
        self.omega0 = None
        self.tc = None
        self.c0 = None
        self.Gm_c03 = None
        self.mod = False
        self.R = None
        self.alpha = None
        self.makeWave = self.__makeWave

    def setMod(self, value):
        if value == True:
            self.mod = True
            self.makeWave = self.__makeModWave
        else:
            self.mod = False
            self.makeWave = self.__makeWave

    def __m(self):
        self.m = self.m1 + self.m2

    def __dm(self):
        self.dm = self.m1 - self.m2

    def __mu(self):
        self.mu = self.m1*self.m2 / self.m

    def __eta(self):
        self.eta = self.mu / self.m
        self.eta2 = self.eta**2

    def __Theta(self, t):
        self.Theta = self.eta / (5.0 * self.Gm_c03 * self.m) *(self.tc - t)

    def __phi(self):
        self.phi = self.phic - 1 / self.eta * ( self.Theta**(5/8.0) + (3715 / 8064.0 + 55 / 96.0 * self.eta) * self.Theta**(3/8.0) - 3 * pi / 4.0 * self.Theta**(1/4.0) + ( 9275495/14450688.0 + 284875 / 258048.0 * self.eta + 1855 / 2048.0 * self.eta2) * self.Theta**(1/8.0) )

    def __omega(self):
        self.omega =  1 / (8.0 * self.Gm_c03 * self.m) * ( self.Theta**(-3/8.0) + (743 / 2688.0 + 11 / 32.0 * self.eta) * self.Theta**(-5/8.) - 3 * pi / 10.0 * self.Theta**(-3/4.0) + (1855099/14450688.0 + 56975 / 258048.0 * self.eta + 371 / 2048.0 * self.eta2) * self.Theta**(-7/8.0) )
    
    def __psi(self):
        self.psi =  self.phi - 2 * self.Gm_c03 * self.m * self.omega * log(self.omega / self.omega0)

    def __x(self):
        self.x = ( self.Gm_c03 * self.m * self.omega ) ** (2/3.)

    def __Hp0(self):
        return - ( 1 + c2 ) * cos( 2 * self.psi )

    def __Hp1_2(self):
        return - s / 8. * self.dm / self.m * ( ( 5 + c2) * cos(self.psi) - 9 * ( 1 + c2 ) * cos(3*self.psi) )

    def __Hp1(self):
        return 1 / 6. * ( ( 19 + 9 * c2 - 2 * c4) - self.eta * ( 19 - 11 * c2 - 6 * c4) ) * cos(2*self.psi) - 4 / 3. * s2 * ( 1 + c2 ) * ( 1 - 3 * self.eta )* cos(4*self.psi) 

    def __Hp3_2(self):
        return s / 192. * self.dm / self.m * ( ( ( 57 + 60 * c2 - c4) - 2 * self.eta * ( 49 - 12 * c2 - c4) ) * cos(self.psi) - 27 / 2. * ( (73 + 40 * c2 - 9 * c4) - 2 * self.eta * ( 25 - 8 * c2 - 9 * c4 ) ) * cos(3*self.psi) + 625 / 2. * ( 1 - 2 * self.eta) * s2 * ( 1 + c2 ) * cos(5*self.psi) ) - 2 * pi * (1 + c2) * cos(2*self.psi)

    def __Hp2(self):
        return 1 / 120. * ( ( 22 + 396 * c2 + 145 * c4 - 5 * c6 ) + 5 / 3. * self.eta * ( 706 - 216 * c2 - 251 * c4 + 15 * c6 )  - 5 * self.eta2 * ( 98 - 108 * c2 + 7 * c4 + 5 * c6 ) ) * cos(2*self.psi) + 2 / 15. * s2 * ( ( 59 + 35 * c2 - 8 * c4 ) - 5 / 3. * self.eta * ( 131 + 59 * c2 - 24 * c4 ) + 5 * self.eta2 * ( 21 - 3 * c2 - 8 * c4 ) ) * cos(4*self.psi) - 81 / 40. * ( 1 - 5 * self.eta + 5 * self.eta2 ) * s4 * ( 1 + c2 ) * cos(6*self.psi) + s / 40. * self.dm / self.m * ( ( 11 + 7 * c2 + 10 * (5 + c2) * ln2 ) * sin(self.psi) - 5 * pi * ( 5 + c2 ) * cos(self.psi) - 27 * ( 7 - 10 * ln3_2) * ( 1 + c2 ) * sin(3*self.psi) + 135 * pi * ( 1 + c2 ) * cos(3*self.psi) ) 

    def __modHp3_2(self):
        return 8 / (3.0 * tan(pi*self.alpha/2)) * (-1/4.0) * ( 1 + c2 ) * sin(2*self.psi)

    def __modHc3_2(self):
        return 8 / (3.0 * tan(pi*self.alpha/2)) * (1/2.0) * c * cos(2*self.psi)
         
    def __modHp2(self):
        return self.dm / (5.0 * self.m * tan(pi*self.alpha/2)) * ( 7/4.0 * ( s2 + ( 1 + c2 ) * cos( 2 * self.psi ) ) * s * sin(self.psi) - 5 * ( s2 - ( 1 + c2 ) * cos(2*self.psi) ) * s * sin(self.psi) + 5 * ( 1 + c2) * sin(2*self.psi) * s * cos(self.phi) )

    def __modHc2(self):
        return self.dm / (5.0 * self.m * tan(pi*self.alpha/2)) * ( 7/2.0 * c * s * sin(2*self.psi) * sin(self.psi) + 10 * c * s * sin(2*self.psi) * sin(self.psi) - 10 * c * s * cos(2*self.psi) * cos(self.psi) )

    def __makeWave(self, m1, m2, phic, tc, t):
        #start_time = time.time()
        self.__setVariables(m1, m2, phic, tc, t)
        hp_r = 2 * self.c0 * self.Gm_c03 * self.m * self.eta * self.x * ( self.__Hp0() + self.x**(1/2.) * self.__Hp1_2() + self.x * self.__Hp1() + self.x**(3/2.) * self.__Hp3_2() + self.x**(2) * self.__Hp2() )        

        return hp_r

    def __makeModWave(self, m1, m2, phic, tc, alpha, t):
        self.__setVariables(m1, m2, phic, tc, t)
        self.alpha = float(alpha)
        hp_r = 2 * self.c0 * self.Gm_c03 * self.m * self.eta * self.x * ( self.__Hp0() + self.x**(1/2.) * self.__Hp1_2() + self.x * self.__Hp1() + self.x**(3/2.) * (self.__Hp3_2() + self.__modHp3_2()) + self.x**(2) * (self.__Hp2() + self.__modHp2()) )        

        if len(hp_r) == 0:
            print "len(wavelength) == 0 for ", m1, m2, phic, tc, alpha  

        return hp_r

    def madeMod(self, m1, m2, phic, tc, alpha, t):
        self.__setVariables(m1, m2, phic, tc, t)
        self.alpha = alpha
        return 2 * self.c0 * self.Gm_c03 * self.m * self.eta * self.x * (  self.x**(3/2.) *  self.__modHp3_2() + self.x**(2) * self.__modHp2() )        

    def __setVariables(self, m1, m2, phic, tc, t):
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.phic = float(phic)
        self.tc = float(tc)
        self.omega0 = 10 * pi
        self.__m()
        #print "\nm", self.m
        self.__dm()
        #print "dm", self.dm
        self.__mu()
        #print "mu", self.mu
        self.__eta()
        #print "eta", self.eta
        self.__Theta(t)
        #print "\nTheta\n", self.Theta
        self.__phi()
        #print "phi", self.phi
        self.__omega()
        #print "\nomega\n", self.omega
        self.__psi()
        #print "psi", self.psi
        self.__x()
        #print "x", self.x






























