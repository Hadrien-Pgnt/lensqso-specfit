import numpy
from numpy.polynomial.hermite import hermval
from scipy import interpolate as I

def makeContinuumLine(lamRest,contDictionary,center):
    ###can make this better.... put the center in the initialization instead.
    #center = 5012
    #slope,amplitude = contDictionary

    slope = contDictionary['slope']
    amp = contDictionary['amp']


    continuum = (center-lamRest)*slope + amp

    continuum[continuum<0] = 0
    return continuum

'''def makeContinuumPL(lamRest,contDictionary,center):
    ###can make this better.... put the center in the initialization instead.
    #center = 5012
    #slope,amplitude = contDictionary

    slope = contDictionary['slope']
    amp = contDictionary['amp']


    continuum = (center-lamRest)*slope + amp

    continuum[continuum<0] = 0
    return continuum
'''
def makeHermite(lamRest,lineDictionary,lam_o):

    #xo, width, relAmplitudes, totalAmplitude = line
    totalAmplitude = lineDictionary['amp']
    width = lineDictionary['width']
    relAmplitudes = lineDictionary['compAmps'].copy()
    dlam = lineDictionary['dlam']

    lam_cen = lam_o-dlam
    #print('lam cen')
    #print(lam_cen)
    if relAmplitudes == []:
        relAmplitudes = [1]


    lam = (lamRest-lam_cen)/(2**0.5*width)
    #print('relAmplitudes')
    #print(relAmplitudes)
    gauss = 1/(2*numpy.pi*width**2)**0.5 * numpy.exp(-1*lam**2)

    #import pylab
    #pylab.plot(lamRest,gauss)
    #pylab.show()
    hermite = totalAmplitude*hermval(lam,relAmplitudes) * gauss
    #import pylab
    #pylab.plot(lamRest, hermite)
    #pylab.show()

    return hermite

#def makeOIII(lamRest,lineDictionary):
#    #xo, width, relAmplitudes, totalAmplitude = line
#    totalAmplitude = lineDictionary['amp']
#    width = lineDictionary['width']
##    relAmplitudes = lineDictionary['compAmps']
#    dlam = lineDictionary['dlam']
#    if relAmplitudes is None:
#        relAmplitudes = [1]


#    lam1 = (lamRest-dlam)/width**2


###Fe needs xo, relative component amplitudes,velocity
#iron = agn_spec_features.makeIron(self.lamRest, Fe)

def makeIron(lamRest,FeLineDictionary, FeDictionaries):
    totalAmp = FeLineDictionary['amp']
    velocity = FeLineDictionary['velocity']
    #print('velocity')
    #print(velocity)
    G = FeLineDictionary['G']
    IZW1 = FeLineDictionary['IZw1']
    S = FeLineDictionary['S']

    dlam = FeLineDictionary['dlam']



    #xo, componentNames,totalAmp, relAmps, velocity = Fe

    lamFes = lamRest-dlam
    modVelocity = int(numpy.around(velocity, decimals=-1))
    componentIndex = 0
    #oFe = numpy.zeros(lamFes.size)
    #for componentName in componentNames:

    feModel = FeDictionaries['F'][modVelocity]
    ffes_F = I.splev(lamFes, feModel)

    feModel = FeDictionaries['G'][modVelocity]
    ffes_G = G*I.splev(lamFes, feModel)

    feModel = FeDictionaries['IZw1'][modVelocity]
    ffes_zw = IZW1*I.splev(lamFes, feModel)


    feModel = FeDictionaries['S'][modVelocity]
    ffes_S = S * I.splev(lamFes, feModel)

    #oFe +=  ffes*relAmps[componentIndex]

    #componentIndex+=1

    ofe = totalAmp*(ffes_F+ffes_G+ffes_zw+ffes_S)
    #import pylab
    #pylab.plot(ofe)
    #pylab.show()
    return ofe



