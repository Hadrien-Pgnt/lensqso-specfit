__author__ = "hpaugnat"

import os, copy
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import eval_hermite, factorial, factorial2
from scipy import interpolate as interp
from scipy.special import voigt_profile
import pandas as pd
import pickle

c = 3e5 #speed of light in km/s

### Rest frame wavelengths (in vacuum) of narrow-line doublets (in Å)
### plus theoretical line-intensity or transition probability ratio (line2/line1), with references)
QSO_narrow_doublets = {'NeV': [(3346.79, 3426.85), 2.73], # Cleri et al. (2023), line-intensity ratio
                       'NeIII': [(3869.86, 3968.59), 1/3.32], # Iwamuro et al. (2003), line-intensity ratio
                       'OIII': [(4960.30, 5008.24), 2.98], # Storey & Zeippen (2000), line-intensity ratio
                       'OI': [(6302.05, 6365.54), 1/3], # # Storey & Zeippen (2000) / Sharpee & Slanger (2006) / Izotov & Thuan (2007) / Revalski et al. (2024), line-intensity ratio
                       'NII': [(6549.86, 6585.27), 1/2.96], #Galavis et al. (1997) / Dojčinović et al. (2023) / Revalski et al. (2024)
                       'SII': [(6718.29, 6732.68), 1/1.2], 
                       # (0.42–1.47, Revalski et al. 2024) - dependence on  electron density & temperature in the NLR, e.g. see Xu, Komossa & Zhou (2008) or Osterbrock (1989) p. 112 -> should be left as a free parameter
                       'SIII': [(9071.1, 9533.2), 2.48], # Revalski et al. (2024), transition probability ratio (2.59 in Podobedova et al. 2009)
                       }

### Rest frame wavelengths (in vacuum) of QSO emission lines (mostly broad lines), in Å
QSO_single_lines = {'Halpha': 6564.7,
                    'Hbeta': 4862.7,
                    'Pa_eps': 9548.6, #HI (8->3)
                    'Heps': 3971.2,
                    'Pa_zeta': 9231.6, #HI (9->3)
                    'Pa_eta': 9017.4, #HI (10->3)
                    'NII_8986': 8986.18,
                    'OII': 3728.48, #narrow line (in reality a doublet but two lines are unresolved), cf Vanden Berk et al. (2001) 
                    }

class SpectrumComponent():
    '''Generic spectrum component.'''

    #name of all the fitted parameters
    linear_params = []
    nonlinear_params = []

    def __init__(self):
        ### Need to define default prior for non-linear parameters
        self.priors = {} #dictionary of the form {param: [initial value, initial dispersion of samples, lower bound, higher bound]}

    def set_priors(self, priors):
        '''Update prior with a dictionary'''
        for param in priors:
            if param in self.nonlinear_params:
                self.priors[param] = priors[param]
            else:
                print('Warning:\'' + param + '\'  is not part of the non-linear parameters used in class '+ 
                      self.__class__.__name__ +', so the prior could not be updated.' )

    def make_flux(lamRest):
        '''Return flux values for that component at all the wavelengths in lamRest.'''
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

class GaussHermiteLine(SpectrumComponent):
    '''Represents an individual emission line, with Gauss-Hermite functions to represent beyond-Gaussian line profiles.
    
        Fixed properties:
            * lambda_rest: rest-frame central wavelength of the emission line  
            * degree: leading order of polynomial in Hermite series (Hermite-Gaussian functions with orders in [0, degree] will be included). For a simple Gaussian line profile, use degree=0.
            * broad: if True, will initialize the parameters and priors like a broad line (and otherwise like a narrow line).
            * fix_c1c2_to_zero: if True, the coefficient of order 1 and 2 will be fixed to 0 and not considered in the array of free parameters (since usually only orders 0,3, and 4 are considered).
            
        Fitted parameters (linear):
            * coeffsHermite: coefficients of the Hermite series, ordered from the lowest to the highest order (eg. the leading order is at index -1). The Gauss-Hermite functions are normalized such that the integrated flux in the line is coeffsHermite[0].
            
        Fitted parameters (non-linear):
            * dlam: shift of the line center relative to *lambda_rest*
            * width: Gaussian rms width (FWHM = 2*sqrt(2*ln(2)) * width)'''

    linear_params = ['coeffsHermite']
    nonlinear_params = ['dlam', 'width']
    
    def __init__(self, lambda_rest, broad=True, degree=0, fix_c1c2_to_zero=True):
        self.lambda_rest = lambda_rest
        self.degree = degree
        self.fix_c1c2_to_zero = fix_c1c2_to_zero
        self.priors = {'dlam': [0, 1, -40, 40]} 
        self.normcoeffs = np.array([1/np.sqrt(2**i *factorial(i)*np.sqrt(np.pi)) for i in range(degree+1)]) 
            #normalization coefficients to have the Hermite-Gaussian functions form an orthonormal basis
        
        if broad: #initialize like a broad line (1000 km/s <~ Doppler FHWM <~ 10000 km/s)
            self.priors['width'] = np.array([3000, 1000, 1000, 10000]) * lambda_rest/c /(2*np.sqrt(2*np.log(2))) 
        else:
            #initialize like a narrow line (100 km/s <~ Doppler FHWM <~ 1000 km/s)
            self.priors['width'] = np.array([400, 100, 100, 1000]) * lambda_rest/c /(2*np.sqrt(2*np.log(2))) 

    def make_flux(self, lamRest, dlam, width, coeffsHermite):
        
        lambda_c = self.lambda_rest + dlam
        lamNorm = (lamRest-lambda_c)/ width #normalized wavelength array
        
        #Use the physicist's Hermite polynomials, and normalize the Hermite-Gaussian functions in order to have them form an orthonormal basis, i.e., integral(psi_n*psi_m) = delta(m,n)
        #then normalize the series such that the first term reduces to a normalized Gaussian 
        
        if not(self.fix_c1c2_to_zero):  #c1, c2 are free parameters so are already in coeffsHermite
            return 1/ (np.sqrt(2*np.sqrt(np.pi))*width) * hermval(lamNorm, np.array(coeffsHermite)*self.normcoeffs) * np.exp(-lamNorm**2/2)
        else: #fix c1=c2=0 -> coeffsHermite only includes c0, c3, ...
            coeffsHermite_extended = np.hstack([coeffsHermite[0], [0,0], coeffsHermite[1:]]) 
            return 1/ (np.sqrt(2*np.sqrt(np.pi))*width) * hermval(lamNorm, coeffsHermite_extended*self.normcoeffs) * np.exp(-lamNorm**2/2)

    def make_transposed_response_matrix(self, lamRest, dlam, width):
        resp_M_t = []
        lambda_c = self.lambda_rest + dlam
        lamNorm = (lamRest-lambda_c)/width #normalized wavelength array
        monomial = np.zeros(self.degree+1)
        for i in range(self.degree+1):      
            monomial *=0
            monomial[i] = 1/np.sqrt(2**i *factorial(i)*np.sqrt(np.pi)) #normalize Hermite-Gaussian functions as an orthonormal basis set
            if not(self.fix_c1c2_to_zero) or not(1<=i<=2):
                resp_M_t.append( 1/ (np.sqrt(2*np.sqrt(np.pi))*width) * hermval(lamNorm, monomial) * np.exp(-lamNorm**2/2))
        return resp_M_t 

    def get_total_flux(self, coeffsHermite, **kwargs_values):
        '''Returns the integrated line flux of a line represented by a Gauss-Hermite series with coefficients coeffsHermite = [c0, c1, c2,...]
        '''
        total_flux = coeffsHermite[0] #count order 0 separately (scipy convention is factorial2(-1)=0)
        for i in range(1, len(coeffsHermite)):
            if (i%2) == 0 : #only even orders matter
                total_flux += coeffsHermite[i]*factorial2(i-1)/np.sqrt(factorial(i))
        return total_flux


    def make_flux_old(self, lamRest, dlam, width, coeffsHermite):
        '''DEPRECATED. Old formulation (not the usual Gauss-Hermite functions)'''
        
        lambda_c = self.lambda_rest + dlam
        lamNorm = (lamRest-lambda_c)/(np.sqrt(2)*width) #wavelength array normalized by Gaussian profile
        
        #Use the physicist's Hermite polynomials, and normalize the Hermite-Gaussian functions in order to have the integral = coeffsHermite[0] for order 0  - such that it is the integrated flux in the line (the integral is =0 for higher orders)
        
        if not(self.fix_c1c2_to_zero):  #c1, c2 are free parameters so are already in coeffsHermite
            return 1 / (np.sqrt(2*np.pi)*width) * hermval(lamNorm, coeffsHermite) * np.exp(-lamNorm**2)
        else: #fix c1=c2=0 -> coeffsHermite only includes c0, c3, ...
            coeffsHermite_extended = np.hstack([coeffsHermite[0], [0,0], coeffsHermite[1:]]) 
            return 1 / (np.sqrt(2*np.pi)*width) * hermval(lamNorm, coeffsHermite_extended) * np.exp(-lamNorm**2)

    def make_transposed_response_matrix_old(self, lamRest, dlam, width):
        '''DEPRECATED. Old formulation (not the usual Gauss-Hermite functions)'''
        resp_M_t = []
        lambda_c = self.lambda_rest + dlam
        lamNorm = (lamRest-lambda_c)/(np.sqrt(2)*width) #wavelength array normalized by Gaussian profile
        monomial = np.zeros(self.degree+1)
        for i in range(self.degree+1):      
            monomial *=0
            monomial[i] = 1
            if not(self.fix_c1c2_to_zero) or not(1<=i<=2):
                resp_M_t.append( 1/(np.sqrt(2*np.pi)*width) * hermval(lamNorm, monomial) * np.exp(-lamNorm**2))
        return resp_M_t


class GaussHermiteLineRelShape(GaussHermiteLine):
    '''Same as GaussHermiteLine, but this time the Hermite coefficients with order >=1 (in relcoeffsHermite) are treated as non-linear parameters (expressed relative to order 0) to allow for the shape parameters to be shared between different images. Only the overall scaling *amp* (i.e., the order 0 coefficient) is treated as a linear parameter.
    '''
    
    def __init__(self, lambda_rest, broad=True, degree=0, fix_c1c2_to_zero=True):
        super().__init__(lambda_rest, broad, degree, fix_c1c2_to_zero)
        self.linear_params = ['amp']
        self.nonlinear_params = ['dlam', 'width', 'relcoeffsHermite']
        self.priors['relcoeffsHermite'] = [0, 0.1, -1, 1] 
        
    def make_flux(self, lamRest, amp, dlam, width, relcoeffsHermite):
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1) 
        return amp*(super().make_flux(lamRest, dlam, width, Hermcoeffs)) #then multiply everything by c_0=amp

    def make_transposed_response_matrix(self, lamRest, dlam, width, relcoeffsHermite):
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1)  
        return [super().make_flux(lamRest, dlam, width, Hermcoeffs)]

    def get_total_flux(self, amp, relcoeffsHermite, **kwargs_values):
        '''Returns the integrated line flux of a line represented by a Gauss-Hermite series with coefficients [c0, c1, c2,...]
        where amp=c0 and relcoeffsHermite = [c1/c0, c2/c0, ...]
        '''        
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1) 
        return amp*(super().get_total_flux(Hermcoeffs)) #then multiply everything by c_0=amp

class VoigtLine(SpectrumComponent):
    '''Represents an individual emission line with a Voigt profile. 
    
    The FHWM of the full profile can be approximated as FWHM_v ≈ sqrt(f_L^2/4+f_G^2)+f_L/2,
    where f_L = 2*gamma is the FHWM of the Lorentzian component and f_G = 2*sqrt(2*ln(2)) * width is the FHWM of the Gaussian component.
    Assuming that f_L ~ f_G, we have gamma ≈ FWHM_v/(sqrt(2)+1) and width ≈ FWHM_v/((sqrt(2)+1)*2*sqrt(2*ln(2))).
        
        Fixed parameters:
            * lambda_rest: rest-frame central wavelength of the emission line  

        Fitted parameters (linear):
            * amp: scaling parameter (the Voigt function will be normalized such that this is the integrated flux in the line)
            
        Fitted parameters (non-linear):
            * dlam: shift of the line center relative to *lambda_rest*
            * width: rms width of the Gaussian component
            * gamma: scale parameter of the Lorentzian component.'''

    linear_params = ['amp']
    nonlinear_params = ['dlam', 'width', 'gamma']
    
    def __init__(self, lambda_rest, broad=True):
        self.lambda_rest = lambda_rest
        self.priors = {'dlam': [0, 1, -40, 40]} 
        if broad: #initialize like a broad line (1000 km/s <~ FHWM <~ 10000 km/s)
            self.priors['width'] = np.array([3000, 1000, 1000, 10000]) * lambda_rest/c /(2*np.sqrt(2*np.log(2)) * (np.sqrt(2)+1)) 
            self.priors['gamma'] = np.array([3000, 1000, 1000, 10000]) * lambda_rest/c / (np.sqrt(2)+1)
        else:
            #initialize like a narrow line (100 km/s <~ FHWM <~ 1000 km/s)
            self.priors['width'] = np.array([400, 100, 100, 1000]) * lambda_rest/c /(2*np.sqrt(2*np.log(2)) * (np.sqrt(2)+1)) 
            self.priors['gamma'] = np.array([400, 100, 100, 1000]) * lambda_rest/c / (np.sqrt(2)+1)
            
    def make_flux(self, lamRest, dlam, width, gamma, amp):
        lambda_c = self.lambda_rest + dlam
        return amp*voigt_profile(lamRest-lambda_c, width, gamma)
    
    def make_transposed_response_matrix(self, lamRest, dlam, width, gamma):
        return [self.make_flux(lamRest, dlam, width, gamma, amp=1)]
    
    def get_total_flux(self, amp, **kwargs_values):
        return amp ##the Voigt profile is normalized such that amp is the total flux in the line
                             

def SingleLine(name, type, broad=True, lambda_rest=None, kwargs_init={}):
    '''Choose the class GaussHermiteLine(), GaussHermiteLineRelShape(), or VoigtLine() to represent a singlet in the dictionary of known QSO lines. 
    Broad line by default. 
    *kwargs_init* can be used to specify {'degree':int, 'fix_c1c2_to_zero':bool} for the GaussHermiteLine() instances
    Can be a generic line at *lambda_rest* if not in the dictionary of known QSO lines (in which case name needs to start with 'LineAt').
    '''
    if not(name.startswith('LineAt')):
        lambda_rest = QSO_single_lines[name]
    
    if type=='GaussHermite':
        return GaussHermiteLine(lambda_rest, broad=broad, **kwargs_init)
    elif type=='GaussHermiteRelShape':
        return GaussHermiteLineRelShape(lambda_rest, broad=broad, **kwargs_init)
    elif type=='Voigt':
        return VoigtLine(lambda_rest, broad=broad)
    else:
        raise TypeError("Type \'" + str(type) + "\' is not accepted by SingleLine(). The input type must either be 'GaussHermite' or 'Voigt'.") 
        

class NarrowDoublet(SpectrumComponent):
    '''Uses two instances of either GaussHermiteLine() or VoigtLine() to represent a narrow-line doublet with the same profile for the two lines, scaled by a predicted ratio.
    Uses the wavelengths and theoretical line ratios listed in the dictionary *QSO_narrow_doublets*
    For type='GaussHermite', the coefficient of the Hermite series are all treated as linear parameters, so they cannot be shared between different images. For type='GaussHermiteRelShape', the Hermite coefficients with order >=1 (in relcoeffsHermite) are treated as non-linear parameters (expressed relative to order 0), so they can be shared.
    *kwargs_init* can be used to specify {'degree':int, 'fix_c1c2_to_zero':bool} for the GaussHermiteLine() instances
    '''
    
    def __init__(self, name, type, kwargs_init={}):
        wavelengths, lineratio = QSO_narrow_doublets[name]

        if type=='GaussHermite':
            self.line1 = GaussHermiteLine(wavelengths[0], broad=False, **kwargs_init) #initialize as narrow-line
            self.line2 = GaussHermiteLine(wavelengths[1], broad=False, **kwargs_init) #initialize as narrow-line
            self.linear_params = ['coeffsHermite']
            self.nonlinear_params = ['dlam', 'width']

        elif type=='GaussHermiteRelShape':
            self.line1 = GaussHermiteLineRelShape(wavelengths[0], broad=False, **kwargs_init) #initialize as narrow-line
            self.line2 = GaussHermiteLineRelShape(wavelengths[1], broad=False, **kwargs_init) #initialize as narrow-line
            self.linear_params = ['amp']
            self.nonlinear_params = ['dlam', 'width', 'relcoeffsHermite']
            
        elif type=='Voigt':
            self.line1 = VoigtLine(wavelengths[0], broad=False) #initialize as narrow-line
            self.line2 = VoigtLine(wavelengths[1], broad=False) #initialize as narrow-line   
            self.linear_params = ['amp']
            self.nonlinear_params = ['dlam', 'width', 'gamma']
            
        else:
            raise TypeError("Type" + str(type) + "is not accepted by NarrowDoublet(). The input type must either be 'GaussHermite' or 'Voigt'.") 
        
        self.line_ratio = lineratio #amp(line2)/amp(line1)
        self.degree = kwargs_init.get('degree', 0) #default to degree=0 (i.e. Gaussian) when not specified
        self.fix_c1c2_to_zero = kwargs_init.get('fix_c1c2_to_zero', True) #default to True (i.e. fix c1=c2=0 for GaussHermite)
        self.priors = copy.deepcopy(self.line1.priors)
        
    def make_flux(self, lamRest, **kwargs_values): 
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        return self.line1.make_flux(lamRest, **kwargs_values) + self.line_ratio*self.line2.make_flux(lamRest, **kwargs_values)

    def make_transposed_response_matrix(self, lamRest, **kwargs_values):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        resp_M_t_line1 = self.line1.make_transposed_response_matrix(lamRest, **kwargs_values)
        resp_M_t_line2 = self.line2.make_transposed_response_matrix(lamRest, **kwargs_values)
        resp_M_t = []
        for i in range(len(resp_M_t_line1)):
            resp_M_t.append(resp_M_t_line1[i] + self.line_ratio*resp_M_t_line2[i])
        return resp_M_t
    
    def set_priors(self, priors): 
        #update prior of this class and also of the two child instances of GaussHermiteLine() or VoigtLine()
        super().set_priors(priors)
        self.line1.set_priors(priors)
        self.line2.set_priors(priors)

    def get_total_flux(self,  **kwargs_values):
        return (1+self.line_ratio)*self.line1.get_total_flux(**kwargs_values)

class NarrowDoubletGaussHermiteRelShape(NarrowDoublet):
    '''DEPRECATED. 
    Same as NarrowDoublet(type='GaussHermite'), but this time the Hermite coefficients with order >=1 (in relcoeffsHermite) are treated as non-linear parameters (expressed relative to order 0) to allow for the shape parameters to be shared between different images. Only the overall scaling *amp* (i.e., the order 0 coefficient) is treated as a linear parameter.
    '''
    
    def __init__(self, name, kwargs_init={}):
        super().__init__(name, 'GaussHermite', kwargs_init)
        self.linear_params = ['amp']
        self.nonlinear_params = ['dlam', 'width', 'relcoeffsHermite']
        self.priors['relcoeffsHermite'] = [0, 0.1, -1, 1] 
        self.fix_c1c2_to_zero = self.line1.fix_c1c2_to_zero
        
    def make_flux(self, lamRest, amp, dlam, width, relcoeffsHermite):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1) only 
                                                        #(c1=c2=0 will be added by the calls to line1 and line2)
        return amp*(self.line1.make_flux(lamRest, dlam, width, Hermcoeffs) + self.line_ratio*self.line2.make_flux(lamRest, dlam, width, Hermcoeffs))

    def make_transposed_response_matrix(self, lamRest, dlam, width, relcoeffsHermite):
        #assume the same line profile for both lines in the doublet, but with an amplitude scaled by the line ratio
        Hermcoeffs = np.hstack([[1], relcoeffsHermite]) #add the 0th order coefficient (=1) only 
                                                    #(c1=c2=0 will be added by the calls to line1 and line2)
        return [self.line1.make_flux(lamRest, dlam, width, Hermcoeffs) + self.line_ratio*self.line2.make_flux(lamRest, dlam, width, Hermcoeffs)]
    
    def set_priors(self, priors): 
        #update prior of this class and also of the two child instances of Line()
        super().set_priors(priors)
        priors_copy = copy.deepcopy(priors)
        priors_copy.pop('relcoeffsHermite', None) #remove coeffsHermite from dictionary to avoid raising an error (not a parameter in Line())
        self.line1.set_priors(priors)
        self.line2.set_priors(priors)


class FeII_Vis_TemplateLines(SpectrumComponent):
    ''' Uses a template from Kovačević et al. (2010) to represent the optical/visible FeII emission lines (4400–5400 Å) with several groups.

    Fitted parameters (linear):
        * 'F', 'G', 'IZw1', 'S': amplitude of each group of lines, see Kovačević et al. (2010)
    Fitted parameters (non-linear):
        * dlam: shift in the wavelength with respect to rest-frame template
        * velocity: Doppler width (in km/s) of the FeII emission region (sets the line widths)
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


class FeII_Vis_TemplateLines_RelAmps(FeII_Vis_TemplateLines):
    ''' Same as FeII_Vis_TemplateLines(), but this time the amplitude coefficients are treated as non-linear parameters (expressed relative to group F) to allow for the ratios to be shared between different images. Only the amplitude *F* of one of the groups of lines is treated as a linear coefficient.
        
    Fitted parameters (linear):
        * F:  amplitude for the group of lines labeled F in Kovačević et al. (2010)
    Fitted parameters (non-linear):
        * dlam: shift in the wavelength with respect to rest-frame template
        * velocity: Doppler width (in km/s) of the FeII emission region (sets the line widths)
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


class FeIIMgII_NIR_TemplateLines(SpectrumComponent):
    ''' Uses a semi-empirical template from Garcia-Rissmann et al. (2012) to represent the near-IR FeII+MgII emission lines (8300–11600 Å), derived from the observed spectrum of IZw1.

    Fixed parameters:
        * range_lambda: only lines in this wavelength range will be considered (to avoid including lines that are too far from the region of interest).

    Fitted parameters (linear):
        * amp: overall scaling of the template
    Fitted parameters (non-linear):
        * dlam: shift in the wavelength with respect to rest-frame template
        * dispVel: velocity dispersion in the FeII emission region (sets the line widths) - related to line profile with dispVel = c*sigma_lambda/lambda_c
    '''

    linear_params = ['amp']
    nonlinear_params = ['dlam', 'dispVel'] 

    def __init__(self, range_lambda=(8900, 9700)):
        self.priors = {'dispVel': [700, 200, 200, 2000], 'dlam': [0, 1, -40, 40]}
        template = pd.read_fwf(os.path.join(os.path.dirname(os.path.realpath(__file__)),'FeII+MgII_template_8300_11600.txt'))
        self.template = template[(template['lambda_air']>range_lambda[0]) & (template['lambda_air']<range_lambda[1])]
        self.nlines = len(template['lambda_air'])
        self.linelambdas = np.expand_dims(template['lambda_air'], axis=1)
        self.rel_amps = template['l_deconv']

    def make_flux(self, lamRest, amp, dlam, dispVel):
        sigma_lambda = dispVel*(self.linelambdas+dlam)/c
        lamNorm = (np.tile(lamRest, (self.nlines, 1)) - self.linelambdas - dlam)/ (np.sqrt(2) * sigma_lambda)
        return amp * np.dot(self.rel_amps, np.exp(-lamNorm**2)/(np.sqrt(2*np.pi)*sigma_lambda)) #sum all line contributions with the correct relative amplitudes
        
    def make_transposed_response_matrix(self, lamRest, dlam, dispVel):
        sigma_lambda = dispVel*(self.linelambdas+dlam)/c
        lamNorm = (np.tile(lamRest, (self.nlines, 1)) - self.linelambdas - dlam)/ (np.sqrt(2) * sigma_lambda)
        return [np.dot(self.rel_amps, np.exp(-lamNorm**2)/(np.sqrt(2*np.pi)*sigma_lambda))] #sum all line contributions with the correct relative amplitudes
                




