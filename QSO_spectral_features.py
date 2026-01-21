__author__ = "hpaugnat"

import os, copy
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
        #define default prior for non-linear parameters
        self.priors = {} #dictionary of the form {param: [initial value, initial dispersion of samples, lower bound, higher bound]}

    def set_priors(self, priors): #update prior with a dictionary
        for param in priors:
            if param in self.nonlinear_params:
                self.priors[param] = priors[param]
            else:
                print('Warning:\'' + param + '\'  is not part of the non-linear parameters used in class '+ 
                      self.__class__.__name__ +', so the prior could not be updated.' )

    def make_flux(lamRest):
        return np.zeros_like(lamRest)

    def make_transposed_response_matrix(self, lamRest):
        '''Returns the transposed linear response matrix, i.e. M^T, where the matrix M of size (N_lam x N_lin) is such that (M*X)^T is the simulated flux of this component, where X^T is the vector with linear parameters (of length N_lin) and N_lam is the number of data points.
        Result should be in the form of a list of length N_lin, with each entry an array of size N_lam.'''
        return []
    
        
class PolynomialContinuum(SpectrumComponent):
    ''' Fixed parameters:
            * lambda_c: center of wavelength range, used as reference wavelength
            * lambda_scale: half-width of wavelength range, used to scale polynomial
            * degree: order of polynomial (total number of coefficients = 1+degree). For a simple linear law, use degree=1.
        Fitted parameters (linear):
            * coeffs: array of len *degree* with values of polynomial coefficients (from highest to lowest order, e.g. the leading order is at index 0).'''

    linear_params = ['coeffs']
    nonlinear_params = []

    def __init__(self, lambda_c, lambda_scale, degree=1):
        self.lambda_c = lambda_c
        self.lambda_scale = lambda_scale
        self.degree = degree
        self.priors = {} 

    def make_flux(self, lamRest, coeffs):
        return np.polyval(coeffs, (lamRest-self.lambda_c)/self.lambda_scale)

    def make_transposed_response_matrix(self, lamRest):
        vandermonde = np.polynomial.polynomial.polyvander((lamRest-self.lambda_c)/self.lambda_scale, self.degree) 
            #in this case the response matrix is the Vandermonde matrix (but need to be careful about coefficient order !)
        resp_M_t = []
        for i in range(self.degree+1):
            resp_M_t.append(vandermonde[:,self.degree-i])
        return resp_M_t 

class PowerLawContinuum(SpectrumComponent):
    ''' Fixed parameters:
            * lambda_c: center of wavelength range, used as reference wavelength
        Fitted parameters (linear):
            * amp: amplitude at lambda_c
        Fitted parameters (non-linear):
            * beta: power-law index'''

    linear_params = ['amp']
    nonlinear_params = ['beta']
    
    def __init__(self, lambda_c):
        self.lambda_c = lambda_c
        self.priors = {'beta': [0,1, -10, 10]}

    def make_flux(self, lamRest, amp, beta):
        return amp*(lamRest/self.lambda_c)**beta

    def make_transposed_response_matrix(self, lamRest, beta):
        return [(lamRest/self.lambda_c)**beta]

class Line(SpectrumComponent):
    '''Represents an individual emission line, with Gauss-Hermite functions to represent beyond-Gaussian line profiles.
        Fixed parameters:
            * lambda_rest: rest-frame central wavelength of the emission line  
            * degree: leading order of polynomial in Hermite series (Hermite-Gaussian functions with orders in [0, degree] will be included). For a simple Gaussian line profile, use degree=0.
        Fitted parameters (linear):
            * coeffsHermite: coefficients of the Hermite series, ordered from the lowest to the highest order (eg. the leading order is at index -1).
        Fitted parameters (non-linear):
            * dlam: shift of the line center relative to *lambda_rest*
            * width: Gaussian rms width (FWHM = 2*sqrt(2*ln(2)) * width)'''

    linear_params = ['coeffsHermite']
    nonlinear_params = ['dlam', 'width']
    
    def __init__(self, lambda_rest, degree, broad=True):
        self.lambda_rest = lambda_rest
        self.degree = degree
        self.priors = {'dlam': [0, 1, -40, 40]} 
        if broad:
            self.priors['width'] = [50, 10, 20, 400] #initialize like a broad line (3000 km/s < gas velocity < 10000 km/s)
        else:
            self.priors['width'] = [15, 3, 1, 50] #initialize like a narrow line (300 km/s < gas velocity < 1000 km/s)      

    def make_flux(self, lamRest, dlam, width, coeffsHermite):
        lambda_c = self.lambda_rest + dlam
        lamNorm = (lamRest-lambda_c)/(np.sqrt(2)*width) #wavelength array normalized by Gaussian profile
        #Use the physicist's Hermite polynomials, and normalize the Hermite-Gaussian functions in order to have the integral = coeffsHermite[0] for order 0  - such that it is the integrated flux in the line (the integral is =0 for higher orders)
        return 1 / (np.sqrt(2*np.pi)*width) * hermval(lamNorm, coeffsHermite) * np.exp(-lamNorm**2)

    def make_transposed_response_matrix(self, lamRest, dlam, width, scale=1):
        resp_M_t = []
        lambda_c = self.lambda_rest + dlam
        lamNorm = (lamRest-lambda_c)/(np.sqrt(2)*width) #wavelength array normalized by Gaussian profile
        monomial = np.zeros(self.degree+1)
        for i in range(self.degree+1):      
            monomial *=0
            monomial[i] = 1
            resp_M_t.append(scale / (np.sqrt(2*np.pi)*width) * hermval(lamNorm, monomial) * np.exp(-lamNorm**2))
        return resp_M_t
                             
class SingleLine(Line):
    '''Wraps the class Line() to represent a singlet in the dictionary of known QSO lines. Broad line by default.'''
    
    def __init__(self, name, degree, broad=True):
        super().__init__(QSO_single_lines[name], degree, broad=broad)

class NarrowDoublet(SpectrumComponent):
    '''Uses two instances of the class Line() to represent a narrow-line doublet with the same profile for the two lines, scaled by a predicted ratio.
    Uses the wavelengths and theoretical line ratios listed in the dictionary *QSO_narrow_doublets*
    With this definition the coefficient of the Hermite series are all treated as linear parameters, so they cannot be shared between different images.
    '''
    
    linear_params = ['coeffsHermite']
    nonlinear_params = ['dlam', 'width']
    
    def __init__(self, name, degree):
        wavelengths, lineratio = QSO_narrow_doublets[name]
        self.line1 = Line(wavelengths[0], degree, broad=False) #initialize as narrow-line
        self.line2 = Line(wavelengths[1], degree, broad=False) #initialize as narrow-line
        self.line_ratio = lineratio #amp(line2)/amp(line1)
        self.degree = degree
        self.priors = {'dlam': [0, 1, -40, 40], 'width': [15, 3, 1, 50]} #initialize as narrow line
        
    def make_flux(self, lamRest, dlam, width, coeffsHermite): 
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        return self.line1.make_flux(lamRest, dlam, width, coeffsHermite) + self.line2.make_flux(lamRest, dlam, width, self.line_ratio*coeffsHermite)

    def make_transposed_response_matrix(self, lamRest, dlam, width):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        resp_M_t_line1 = self.line1.make_transposed_response_matrix(lamRest, dlam, width, scale=1)
        resp_M_t_line2 = self.line2.make_transposed_response_matrix(lamRest, dlam, width, scale=self.line_ratio)
        resp_M_t = []
        for i in range(len(resp_M_t_line1)):
            resp_M_t.append(resp_M_t_line1[i] + resp_M_t_line2[i])
        return resp_M_t
    
    def set_priors(self, priors): 
        #update prior of this class and also of the two child instances of Line()
        super().set_priors(priors)
        self.line1.set_priors(priors)
        self.line2.set_priors(priors)

class NarrowDoubletRelShape(NarrowDoublet):
    '''Same as NarrowDoublet(), but this time the Hermite coefficients with order >=1 are treated as non-linear parameters (expressed relative to order 0) to allow for the shape parameters to be shared between different images. Only the overall scaling *amp* is treated as a linear coefficient.
    '''
    
    linear_params = ['amp']
    nonlinear_params = ['dlam', 'width', 'relcoeffsHermite']
    
    def __init__(self, name, degree):
        super().__init__(name, degree)
        self.priors['relcoeffsHermite'] = [0, 0.1, -1, 1] 
        
    def make_flux(self, lamRest, amp, dlam, width, relcoeffsHermite):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1)
        return amp*(self.line1.make_flux(lamRest, dlam, width, Hermcoeffs) + self.line_ratio*self.line2.make_flux(lamRest, dlam, width, Hermcoeffs))

    def make_transposed_response_matrix(self, lamRest, dlam, width, relcoeffsHermite):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1)
        return [self.line1.make_flux(lamRest, dlam, width, Hermcoeffs) + self.line_ratio*self.line2.make_flux(lamRest, dlam, width, Hermcoeffs)]
    
    def set_priors(self, priors): 
        #update prior of this class and also of the two child instances of Line()
        super().set_priors(priors)
        priors_copy = copy.deepcopy(priors)
        priors_copy.pop('relcoeffsHermite', None) #remove coeffsHermite from dictionary to avoid raising an error (not a parameter in Line())
        self.line1.set_priors(priors)
        self.line2.set_priors(priors)


class FeIITemplateLines(SpectrumComponent):
    ''' Uses a template from Kovačević et al. (2010) to represent the optical FeII emission lines (λλ4400–5400) with several group.

    Fitted parameters (linear):
        * 'F', 'G', 'IZw1', 'S': amplitude of each group of lines, see Kovačević et al. (2010)
    Fitted parameters (non-linear):
        * dlam: shift in the wavelength with respect to rest-frame template
        * velocity: velocity dispersion in the FeII emission region (sets the line widths)
    '''
    
    linear_params = ['F', 'G', 'IZw1', 'S']
    nonlinear_params = ['dlam', 'velocity'] 
    path_template = 'FeII_template_4000_5500'
    
    def __init__(self):
        self.FeIItemplate = {}
        self.priors = {'velocity': [3500, 500, 700, 5990], 'dlam': [0, 1, -40, 40]}
        for feName in ['F', 'G', 'IZw1', 'S']:
            path_dict = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.path_template, '%s_modDict_2.pickle' % (feName))
            modDict = pickle.load(open(path_dict, 'rb'))
            self.FeIItemplate[feName] = modDict

    def make_flux(self, lamRest, dlam, velocity, F, G, IZw1, S):
        lamFes = lamRest-dlam
        modVelocity = int(np.around(velocity, decimals=-1))

        feModel = self.FeIItemplate['F'][modVelocity]
        ffes_F = F*interp.splev(lamFes, feModel)
    
        feModel = self.FeIItemplate['G'][modVelocity]
        ffes_G = G*interp.splev(lamFes, feModel)
    
        feModel = self.FeIItemplate['IZw1'][modVelocity]
        ffes_Izw1 = IZw1*interp.splev(lamFes, feModel)
    
        feModel = self.FeIItemplate['S'][modVelocity]
        ffes_S = S * interp.splev(lamFes, feModel)
    
        return ffes_F+ffes_G+ffes_Izw1+ffes_S

    def make_transposed_response_matrix(self, lamRest, dlam, velocity):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        resp_M_t = []
        lamFes = lamRest-dlam
        modVelocity = int(np.around(velocity, decimals=-1))
        for feName in ['F', 'G', 'IZw1', 'S']:
            feModel = self.FeIItemplate[feName][modVelocity]
            resp_M_t.append(interp.splev(lamFes, feModel))
        return resp_M_t


class FeIITemplateLinesRelAmps(FeIITemplateLines):
    ''' Same as FeIITemplateLines(), but this time the amplitude coefficients are treated as non-linear parameters (expressed relative to group F) to allow for the ratios to be shared between different images. Only the amplitude *F* of one of the groups of lines is treated as a linear coefficient.
        
    Fitted parameters (linear):
        * F:  amplitude for the group of lines labeled F in Kovačević et al. (2010)
    Fitted parameters (non-linear):
        * dlam: shift in the wavelength with respect to rest-frame template
        * velocity:  velocity dispersion in the FeII emission region (sets the line widths)
        * relG, relIZw1, relS : ratio of the amplitude of group G (resp., IZw1, and S) to the amplitude of group F, with the labels from Kovačević et al. (2010)
    '''

    linear_params = ['F']
    nonlinear_params = ['dlam', 'velocity', 'relG', 'relIZw1', 'relS']
    
    def __init__(self):
        super().__init__()
        self.priors['relG'] = [0.04, 0.01,0, 10]
        self.priors['relIZw1'] = [0.004, 0.001, 0,10]
        self.priors['relS'] = [0.1, 0.1, 0,10]

    def make_flux(self, lamRest, dlam, velocity, F, relG, relIZw1, relS):
        return F * super().make_flux(lamRest, dlam, velocity, 1, relG, relIZw1, relS)

    def make_transposed_response_matrix(self, lamRest, dlam, velocity, relG, relIZw1, relS):
        resp_M_t_linear = np.array(super().make_transposed_response_matrix(lamRest, dlam, velocity))
        return [np.dot([1, relG, relIZw1, relS], resp_M_t_linear)]        

