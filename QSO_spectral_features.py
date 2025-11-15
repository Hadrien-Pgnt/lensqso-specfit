__author__ = "hpaugnat"

import os
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy import interpolate as interp
import pickle

### Rest frame wavelengths of narrow-line doublets (in Å), 
### plus theoretical line-intensity or transition probability ratio (line2/line1), with references)
QSO_narrow_doublets = {'NeV': [(3345.82, 3425.88), None],
                       'OII': [(3726.03, 3728.815), None],
                       'NeIII': [(3868.76, 3967.47), None],
                       'OIII': [(4958.91, 5006.84), 2.98], # Storey & Zeippen (2000), line-intensity ratio
                       'OI': [(6300.30, 6363.78), None],
                       'NII': [(6548.05, 6583.46), 2.96], #Galavis et al. (1997), Dojčinović et al. (2023)
                       'SII': [(6716.44, 6730.81), None],
                       'SIII': [(9068.6, 9531.1), None],
                       }

### Rest frame wavelengths of QSO emission lines (mostly broad lines), in Å
QSO_single_lines = {'Halpha': 6562.82,
                    'Hbeta': 4861.33,
                    'Pa_eps': 9545.97,
                    'Heps': 3970.08,
                    }

class SpectrumComponent():
    '''Generic spectrum component.'''

    #name of all the fitted parameters
    linear_params = []
    nonlinear_params = []

    def __init__(self):
        #define default prior
        self.priors = {} #dictionary of the form {param: [initial value, initial dispersion of samples, lower bound, higher bound]}

    def set_priors(self, priors): #update prior with a dictionary
        for param in priors:
            if param in self.param_names:
                self.priors[param] = priors[param]
            else:
                print('Warning: the parameter name \'' + param + '\'  is not used in class '+ self.__class__.__name__ +', so the prior could not be updated.' )

    def make_flux(lamRest):
        return np.zeros_like(lamRest)

    def make_response_matrix(self, lamRest):
        '''Returns the transposed linear response matrix, i.e. M^T, where the matrix M of size (N_lam x N_lin) is such that (M*X)^T is the simulated flux of this component, where X^T is the vector with linear parameters (of length N_lin) and N_lam is the number of data points'''
        return []
    
        
class PolynomialContinuum(SpectrumComponent):
    ''' Fixed parameters:
            * lambda_c: center of wavelength range, used as reference wavelength
            * degree: order of polynomial (total number of coefficients = 1+degree). For a simple linear law, use degree=1.
        Fitted parameters (linear):
            * coeffs: array of len *degree* with values of polynomial coefficients (from highest to lowest order, e.g. the leading order is at index 0).'''

    #linear_params = ['coeffs']
    #nonlinear_params = []
    param_names = ['amp', 'coeffs']

    def __init__(self, lambda_c, degree=1):
        self.lambda_c = lambda_c
        self.degree = degree
        self.priors = {'amp': [1,0.1,0,100], #use a different prior for the constant coefficient
                       'coeffs': [0,1, -10, 10]} #use a common prior for all other coeffs

    def make_flux(self, lamRest, coeffs):
        continuum = np.polyval(coeffs, (lamRest-self.lambda_c))*(lamRest-self.lambda_c) + amp
        return continuum

    def make_response_matrix(self, lamRest):
        '''In this case the response matrix is the Vandermonde matrix'''
        return np.polynomial.polynomial.polyvander(x, self.degree)
        

class PowerLawContinuum(SpectrumComponent):
    ''' Fixed parameters:
            * lambda_c: center of wavelength range, used as reference wavelength
        Fitted parameters:
            * amp: amplitude at lambda_c
            * beta: power-law index'''

    param_names = ['amp', 'beta']
    
    def __init__(self, lambda_c):
        self.lambda_c = lambda_c
        self.priors = {'amp': [1,0.1,0,100], 'beta': [0,1, -10, 10]}

    def make_flux(self, lamRest, amp, beta):
        return amp*(lamRest/self.lambda_c)**beta

class Line(SpectrumComponent):
    '''Represents an individual emission line, with Gauss-Hermite functions to represent beyond-Gaussian line profiles.
        Fixed parameters:
            * lambda_rest: rest-frame central wavelength of the emission line  
            * degree: leading order of polynomial in Hermite series (Hermite-Gaussian functions with orders in [0, degree] will be included). For a simple Gaussian line profile, use degree=0.
        Fitted parameters:
            * amp: amplitude of the Gaussian profile (i.e., coefficient in front of H_0 in the Hermite series)
            * dlam: shift of the line center relative to *lambda_rest*
            * width: Gaussian rms width (FWHM = 2*sqrt(2*ln(2)) * width)
            * coeffsHermite: coefficients of the Hermite series with order >0, relative to order 0 (the coefficient in front of H_0 will automatically be set to 1). The coefficients are ordered from the lowest to the highest order (eg. the leading order is at index -1).'''

    param_names = ['amp', 'dlam', 'width', 'coeffsHermite']
    
    def __init__(self, lambda_rest, degree, broad=True):
        self.lambda_rest = lambda_rest
        self.degree = degree
        self.priors = {'amp': [1,0.1,0,100], 'dlam': [0, 1, -40, 40], 'coeffsHermite': [0, 0.1, -1, 1]} #use a common prior for all relative coeffs
        if broad:
            self.priors['width'] = [50, 10, 20, 400] #initialize like a broad line (3000 km/s < gas velocity < 10000 km/s)
        else:
            self.priors['width'] = [15, 3, 1, 50] #initialize like a narrow line (300 km/s < gas velocity < 1000 km/s)
        self.Hermcoeffvalues = np.zeros(1+degree)
        self.Hermcoeffvalues[0] = 1 #set the coefficient in front of H_0 to 1
        

    def make_flux(self, lamRest, amp, dlam, width, coeffsHermite):
        lambda_c = self.lambda_rest + dlam
        self.Hermcoeffvalues[1:] = coeffsHermite
        lamNorm = (lamRest-lambda_c)/(np.sqrt(2)*width) #wavelength array normalized by Gaussian profile
        #Use the physicist's Hermite polynomials, and normalize the Hermite-Gaussian functions in order to have the integral =1 for order 0 and =0 for higher orders - such that amp is the integrated flux in the line regardless of its shape.
        return amp / (np.sqrt(2*np.pi)*width) * hermval(lamNorm,self.Hermcoeffvalues) * np.exp(-lamNorm**2)
                             
        
class SingleLine(Line):
    '''Wraps the class Line() to represent a singlet in the dictionary of known QSO lines. Broad line by default.'''
    
    def __init__(self, name, degree, broad=True):
        super().__init__(QSO_single_lines[name], degree, broad=broad)

class NarrowDoublet(SpectrumComponent):
    '''Uses two instances of the class Line() to represent a narrow-line doublet with the same profile for the two lines, scaled by a predicted ratio.
    Uses the wavelengths and theoretical line ratios listed in the dictionary *QSO_narrow_doublets*
    '''

    param_names = ['amp', 'dlam', 'width', 'coeffsHermite']
    
    def __init__(self, name, degree):
        wavelengths, lineratio = QSO_narrow_doublets[name]
        self.line1 = Line(wavelengths[0], degree, broad=False) #initialize as narrow-line
        self.line2 = Line(wavelengths[1], degree, broad=False) #initialize as narrow-line
        self.line_ratio = lineratio #amp(line2)/amp(line1)
        self.degree = degree
        self.priors = {'amp': [1,0.1,0,100], 'dlam': [0, 1, -40, 40], 'coeffsHermite': [0, 0.1, -1, 1], 'width': [15, 3, 1, 50]} #initialize as narrow line
        
    def make_flux(self, lamRest, amp, dlam, width, coeffsHermite):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        return self.line1.make_flux(lamRest, amp, dlam, width, coeffsHermite) + self.line2.make_flux(lamRest, amp*self.line_ratio, dlam, width, coeffsHermite)

    def set_priors(self, priors): 
        #update prior of this class and also of the two child instances of Line()
        super().set_priors(priors)
        self.line1.set_priors(priors)
        self.line2.set_priors(priors)


class FeIITemplateLines(SpectrumComponent):
    ''' TBD

    Fitted parameters:
        * amp: amplitude scaling parameter
        * dlam: shift in the wavelength with respect to rest-frame template
        * velocity: 
        * G:
        * IZw1:
        * S: 
    '''
    param_names = ['amp', 'dlam', 'velocity', 'G', 'IZw1', 'S'] 
    path_template = 'FeII_template_4000_5500'
    
    def __init__(self):
        self.FeIItemplate = {}
        self.priors = {'velocity': [3500, 500, 1, 5990], 'amp': [1,0.1,0,100], 'dlam': [0, 1, -40, 40],
                             'G':[0.04, 0.01,0, 10],'IZw1':[0.004, 0.001, 0,10],'S':[0.1, 0.1, 0,10]}
        for feName in ['F', 'G', 'IZw1', 'S']:
            path_dict = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.path_template, '%s_modDict_2.pickle' % (feName))
            modDict = pickle.load(open(path_dict, 'rb'))
            self.FeIItemplate[feName] = modDict

    def make_flux(self, lamRest, amp, dlam, velocity, G, IZw1, S):
        lamFes = lamRest-dlam
        modVelocity = int(np.around(velocity, decimals=-1))

        feModel = self.FeIItemplate['F'][modVelocity]
        ffes_F = interp.splev(lamFes, feModel)
    
        feModel = self.FeIItemplate['G'][modVelocity]
        ffes_G = G*interp.splev(lamFes, feModel)
    
        feModel = self.FeIItemplate['IZw1'][modVelocity]
        ffes_zw = IZw1*interp.splev(lamFes, feModel)
    
        feModel = self.FeIItemplate['S'][modVelocity]
        ffes_S = S * interp.splev(lamFes, feModel)
    
        total_Fe_flux = amp*(ffes_F+ffes_G+ffes_zw+ffes_S)
        return total_Fe_flux
        