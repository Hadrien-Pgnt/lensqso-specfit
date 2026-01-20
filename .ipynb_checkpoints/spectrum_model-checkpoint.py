__author__ = "hpaugnat"

import numpy as np
import copy
import matplotlib.pyplot as plt
from cycler import cycler

import QSO_spectral_features as spec_feat

custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#7f7f7f', '#9467bd', '#e377c2', '#bcbd22', '#17becf'])
plt.rcParams['axes.prop_cycle'] = custom_cycler

class CustomError(Exception):
    '''Custom exception raised for specific error scenarios.'''
    def __init__(self, message):
            super().__init__(message) # Call the base class constructor

class QuasarSpectrum():
    '''Docstring TBD.
    *data: list with [rest wavelength array, flux values array, flux uncertainties array] (all need to have the same shape
    *kwargs_model: dictionary of the form {'continuum': 'powerlaw' or ('polynomial', degree), 
                                            'narrow_doublets': list of (name, nb), where 1+nb is the number of Gauss-Hermite polynomials used
                                            'single_lines': list of (name or wavelength, nb, nature), where 1+nb is the number of Gauss-Hermite polynomials used and nature = 'broad'/'narrow' to use a broad/narrow line prior.
                                            'Template_lines': list of names (only FeII available for now)}.
    *narrow_doublet_linHerm: list of narrow_doublet names for which the Hermite series coefficients are treated as linear parameters.
    *FeII_linamps: bool, True if all the FeII line family amplitudes are treated as linear parameters (only F will be otherwise)'''

    def __init__(self, data, kwargs_model, narrow_doublet_linHerm=[], FeII_linamps=True):
        self.lambda_array = data[0]
        self.flux_array = data[1]
        self.flux_err_array = data[2]
        assert np.shape(self.lambda_array == self.flux_array)
        assert np.shape(self.lambda_array == self.flux_err_array)
        central_lambda = (self.lambda_array[0]+self.lambda_array[-1])/2
        
        self.kwargs_model = kwargs_model
        self.feature_dict = {}

        ## Define continuum component
        try:
            if kwargs_model['continuum']=='powerlaw':
                self.feature_dict['continuum'] = spec_feat.PowerLawContinuum(lambda_c=central_lambda)
            elif kwargs_model['continuum'][0]=='polynomial':
                lambda_scale = (self.lambda_array[-1]-self.lambda_array[0])/2
                self.feature_dict['continuum'] = spec_feat.PolynomialContinuum(lambda_c=central_lambda, lambda_scale=lambda_scale,
                                                                               degree=kwargs_model['continuum'][1])
            else:
                raise IndexError
        except IndexError:
            raise CustomError('Error in kwargs_model: the continuum needs to be either \'powerlaw\' or (\'polynomial\', degree)')

        ## Define narrow-line doublets
        for doublet in kwargs_model['narrow_doublets']:
            if doublet[0] in narrow_doublet_linHerm: #Hermite coefficients are included as linear parameters
                self.feature_dict[doublet[0]+'_narrow_doublet'] = spec_feat.NarrowDoublet(name=doublet[0], degree=doublet[1])
            else: #only c_0 (the coefficient in front of H_0) is treated as a linear parameter, the other coefficients are expressed relative to c_0 and are sampled as non-linear parameters
                self.feature_dict[doublet[0]+'_narrow_doublet'] = spec_feat.NarrowDoubletRelShape(name=doublet[0], degree=doublet[1])
                
        ## Define single lines 
        for line in kwargs_model['single_lines']:
            assert (line[2]=='broad' or line[2]=='narrow')
            if isinstance(line[0], str):
                self.feature_dict[line[0]+'_'+ line[2]] = spec_feat.SingleLine(name=line[0], degree=line[1], broad=(line[2]=='broad'))
            else:
                self.feature_dict['Line_at_'+line[0]+'_'+ line[2]] = spec_feat.Line(lambda_rest=line[0], degree=line[1], broad=(line[2]=='broad'))

        ## Define template lines 
        for line in kwargs_model['Template_lines']:
            if line=='FeII':
                if FeII_linamps:
                    self.feature_dict['FeII_template'] = spec_feat.FeIITemplateLines()
                else:
                    self.feature_dict['FeII_template'] = spec_feat.FeIITemplateLinesRelAmps()
            else:
                raise CustomError('Error in kwargs_model: only FeII is supported in Template_lines for the moment.')

        self.lin_param_handler = LinearParamHandler(self.feature_dict)

    def get_priors(self, feature):
        '''Docstring TBD.'''
        return self.feature_dict[feature].priors

    def set_priors(self, feature, priors):
        '''Docstring TBD.'''
        self.feature_dict[feature].set_priors(priors)
        
    def simulateSpectrum(self, kwargs_values, mask_array=None, tol_positive=-1e-3, plot_ax=None):
        '''Docstring TBD.
        mask_array: boolean array indicating which wavelengths should be included (in the likelihood calculation / plot / optimization of linear parameters.)
        tol_positive: consider that fluxes above this value are still positive'''

        if mask_array is None:
            mask_array = np.ones_like(self.lambda_array)
        else:
            assert (mask_array.shape == self.lambda_array.shape)

        sim_spec = np.zeros_like(self.lambda_array)
        check_positive = True #test whether all the individual spectral components have positive flux
        for feature in self.feature_dict:    
            sim_feat = self.feature_dict[feature].make_flux(self.lambda_array, **kwargs_values[feature])
            check_positive *= (np.min(sim_feat*mask_array)>=tol_positive) #change check_positive to False if the flux goes below tol_positive
            sim_spec += sim_feat
            if plot_ax is not None:
                plot_ax.plot(self.lambda_array, sim_feat, label=feature)

        if plot_ax is not None:
            plot_ax.errorbar(self.lambda_array[mask_array>0], self.flux_array[mask_array>0], xerr=None, yerr=self.flux_err_array[mask_array>0], 
                             fmt = '.', label='data', color='k', zorder=0)
            plot_ax.plot(self.lambda_array, sim_spec, label='total', color='#d62728')
            plot_ax.legend()
            plt.xlabel(r'Rest wavelength ($\AA$)')
            plt.ylabel(r'Flux (in input units)' )

        return sim_spec, check_positive 

    def make_transposed_response_matrix(self, kwargs_nonlinear, mask_array=None):
        ''' Docstring TBD.
        kwargs_nonlinear : dictionary containing values for all the non-linear parameters
        mask_array: boolean array indicating which wavelengths should be included in the likelihood calculation and optimization of linear parameters.'''
        
        if mask_array is None:
            mask_array = np.ones_like(self.lambda_array)
        else:
            assert (mask_array.shape == self.lambda_array.shape)
        
        resp_M_t = []
        for feature in self.feature_dict:
            resp_M_t.extend(self.feature_dict[feature].make_transposed_response_matrix(self.lambda_array[mask_array>0], **kwargs_nonlinear[feature]))
        return resp_M_t

    def solve_linear_params(self, kwargs_nonlinear, mask_array=None):
        ''' Docstring TBD. return MLE for linear param vector + expected covariance matrix
        kwargs_nonlinear : dictionary containing values for all the non-linear parameters
        mask_array: boolean array indicating which wavelengths should be included in the likelihood calculation and optimization of linear parameters.'''

        if mask_array is None:
            mask_array = np.ones_like(self.lambda_array)
        else:
            assert (mask_array.shape == self.lambda_array.shape)
        
        resp_M_T = self.make_transposed_response_matrix(kwargs_nonlinear, mask_array=mask_array)
        weight_matrix = np.diag(1/self.flux_err_array[mask_array>0]**2)

        M_T_times_W = resp_M_T @ weight_matrix
        #print(np.shape(resp_M_T), np.shape(weight_matrix), np.shape(M_T_times_W))
        Sigma_X = np.linalg.inv(M_T_times_W @ np.transpose(resp_M_T))

        return np.dot(Sigma_X @ M_T_times_W, self.flux_array[mask_array>0]), Sigma_X
        

    def check_bounds(self, kwargs_nonlinear, verbose=False):
        '''Returns True if all the non-linear parameters within kwargs_values are within the bounds defined in each spectrum component's prior, and False otherwise. verbose=True will print a message indicating which parameter was first found to be outside the bounds, if any.'''
        for feature in self.feature_dict:
            for param_name in self.feature_dict[feature].nonlinear_params:
                if param_name == 'relcoeffsHermite': #multiple parameter values in an array
                    if (np.any(kwargs_nonlinear[feature][param_name] < self.feature_dict[feature].priors[param_name][2]) or 
                        np.any(kwargs_nonlinear[feature][param_name] > self.feature_dict[feature].priors[param_name][3])):                
                        #check if any element of the array is outside the bounds
                        if verbose:
                            print(feature + '_' + param_name + ' is out of prior bounds: value is ', kwargs_nonlinear[feature][param_name], 'but bounds are [',
                                  self.feature_dict[feature].priors[param_name][2], ',', self.feature_dict[feature].priors[param_name][3], ']')
                        return False
                else: #one parameter, single numerical value (NB: polynomial coeffs are necessarily linear so cannot be fixed)
                    if (kwargs_nonlinear[feature][param_name] < self.feature_dict[feature].priors[param_name][2] or
                        kwargs_nonlinear[feature][param_name] > self.feature_dict[feature].priors[param_name][3]):
                        if verbose:
                            print(feature + '_' + param_name + ' is out of prior bounds: value is ', kwargs_nonlinear[feature][param_name], 'but bounds are [',
                                  self.feature_dict[feature].priors[param_name][2], ',', self.feature_dict[feature].priors[param_name][3], ']')
                        return False
        return True

    def log_likelihood(self, kwargs_nonlinear, mask_array=None, tol_positive=-1e-3, check_bounds=True, verbose=False, plot_ax=None):
        '''Docstring TBD.
        mask_array: boolean array indicating which wavelengths should be included in the likelihood calculation and optimization of linear parameters.'''
        
        if mask_array is None:
            mask_array = np.ones_like(self.lambda_array)
        else:
            assert (mask_array.shape == self.lambda_array.shape)
        
        if check_bounds:
            if not(self.check_bounds(kwargs_nonlinear=kwargs_nonlinear, verbose=verbose)): #if one parameter is outside of the bounds
                return -np.inf, None

        X_opt, Sigma_X = self.solve_linear_params(kwargs_nonlinear, mask_array=mask_array)
        kwargs_values = self.lin_param_handler.add_linear_values_to_kwargs(lin_values_array=X_opt, kwargs_nonlinear=kwargs_nonlinear)

        sim_spec, check_positive = self.simulateSpectrum(kwargs_values, mask_array=mask_array, tol_positive=tol_positive, plot_ax=plot_ax)
        if not(check_positive): #if one of the spectral components has negative flux somewhere in the wavelength range
            return -np.inf, kwargs_values

        #log-likelihood after marginalizing over the linear parameters
        log_lik = np.sum(-(sim_spec - self.flux_array)**2/self.flux_err_array**2 * mask_array) + ( 
            len(X_opt)/2 *np.log(2*np.pi) + 1/2 * np.log(np.linalg.det(Sigma_X)) )
        return log_lik, kwargs_values
                
        
class LinearParamHandler():

    def __init__(self, feature_dict):
        self.linear_param_list = [] 
        self.multiplicity_dict = {} #used to store the number of coefficients when the parameters are arrays
        nb_lin_params = 0 # total number of linear parameters
        
        for feature in feature_dict:
            for param_name in feature_dict[feature].linear_params:
                if param_name=='coeffsHermite': #coefficients of Hermite series for Gauss-Hermite function representation: array of parameters, sorted from lowest to highest order, including order 0 when treated as linear parameters
                    self.linear_param_list.extend([feature+'_'+param_name+str(i) for i in range(1+feature_dict[feature].degree)])
                    self.multiplicity_dict[feature] = {'coeffsHermite':1+feature_dict[feature].degree}
                    nb_lin_params += 1+feature_dict[feature].degree
                elif param_name=='coeffs':  #polynomial coefficients: array of parameters, sorted from highest to lowest order, including order 0
                    N = feature_dict[feature].degree
                    self.linear_param_list.extend([feature+'_'+param_name+str(N-i) for i in range(N+1)])
                    self.multiplicity_dict[feature] = {'coeffs':1+feature_dict[feature].degree}
                    nb_lin_params += 1+feature_dict[feature].degree
                else: #single numerical value
                    self.linear_param_list.append(feature+'_'+param_name)
                    nb_lin_params += 1

        self.nb_lin_params = nb_lin_params

    def add_linear_values_to_kwargs(self, lin_values_array, kwargs_nonlinear):
        '''Takes a dictionary of non-linear parameters (kwargs_nonlinear) and an array with values for the linear parameters, sorted appropriately ;
        and returns a dictionary with all the parameters combined.'''
        kwargs_values = copy.deepcopy(kwargs_nonlinear)
        for i in range(len(self.linear_param_list)):
            feature, param_name = self.linear_param_list[i].rsplit('_', 1)
            if param_name.startswith('coeffsHermite'): #coefficients of Hermite series are sorted from lowest to highest order, including order 0 when treated as linear parameters
                coeff_nb = int(param_name.removeprefix('coeffsHermite'))
                if not('coeffsHermite' in kwargs_values[feature]): #create an array to store the coeffs if it does not exist already
                    kwargs_values[feature]['coeffsHermite'] = np.zeros(self.multiplicity_dict[feature]['coeffsHermite'])
                kwargs_values[feature]['coeffsHermite'][coeff_nb] = lin_values_array[i]
            
            elif param_name.startswith('coeffs'): #polynomial coefficients are sorted from highest to lowest order, including order 0
                coeff_nb = int(param_name.removeprefix('coeffs'))
                if not('coeffs' in kwargs_values[feature]): #create an array to store the coeffs if it does not exist already
                    kwargs_values[feature]['coeffs'] = np.zeros(self.multiplicity_dict[feature]['coeffs'])
                N = len(kwargs_values[feature]['coeffs'])-1
                kwargs_values[feature]['coeffs'][N-coeff_nb] = lin_values_array[i]
            else:
                kwargs_values[feature][param_name] = lin_values_array[i]

        return kwargs_values
                    


        