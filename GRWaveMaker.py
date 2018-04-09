
from numpy import cos, sin , log, pi, arange, isnan, isfinite
import time


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

    def __boundaryp(self, R):
        a = 12.0 / 5.0 * self.eta * R 
        print "a", a
        b = self.x**4 * ( 3 * s2 + 5 * (1+c2) * cos ( 2 * self.phi ) ) 
        print "b", b
        c = self.x**5 * ( ( 47 + 5 * self.eta ) * s2 + ( 41 - self.eta ) * ( 1 + c2 ) * cos( 2 * self.phi ) ) 
        print "c", c
        return a*(b+c)    

    def __boundaryc(self, R):
        return 24.0 / 5.0 * self.eta * R * ( 3 * self.x**4 + ( 25 - self.eta ) * self.x**5 )* c * sin( 2 * self.phi ) 
         

    def makeWave(self, m1, m2, phic, tc, t):
        #start_time = time.time()
        self.__setVariables(m1, m2, phic, tc, t)

        hp_r = 2 * self.c0 * self.Gm_c03 * self.m * self.eta * self.x * ( self.__Hp0() + self.x**(1/2.) * self.__Hp1_2() + self.x * self.__Hp1() + self.x**(3/2.) * self.__Hp3_2() + self.x**(2) * self.__Hp2() )        
        #print "\nhp_r\n", hp_r
        
        #hc_r = 2 * self.G * self.m * self.eta / self.c0**2 * self.x * ( self.__Hc0() + self.x**(1/2.) * self.__Hc1_2() + self.x * self.__Hc1() + self.x**(3/2.) * self.__Hc3_2() + self.x**(2) * self.__Hc2() )
        #print "\nhp_r\n", hp_r        

        #print("--- wave made in %s seconds ---" % (time.time() - start_time))

        j = len(hp_r)
        for i in range(len(hp_r)):
            if not isnan(hp_r[i]):
                j = i
                break
        hp_r = hp_r[j:]

        return hp_r

    def __setVariables(self, m1, m2, phic, tc, t):
        self.m1 = m1
        self.m2 = m2
        self.phic = phic
        self.tc = tc
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

    def makeModWave(self, m1, m2, phic, tc, t, R):
        hp_r = makeWave(m1, m2, phic, tc, t)
        hp_r = hp_r + self.__boundaryp(R)

    def makeOnlyMod(self, m1, m2, phic, tc, t, R):
        self.__setVariables(m1, m2, phic, tc, t)
        return self.__boundaryp(R)

if __name__ == "__main__":
    mMaker = GRWaveMaker()
    times = -arange(300)*10 -300
    start_time = time.time()
    hp_r = mMaker.makeWave(m1, m2, phic = 0, omega0 = pi * 10, tc = 0, t = times)
    print("--- %s seconds ---" % (time.time() - start_time))
    print "\n"
    #print hp_r




























