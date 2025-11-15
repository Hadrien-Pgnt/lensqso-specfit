__author__ = "hpaugnat"

import numpy as np
import QSO_spectral_features as spec_feat
import copy
import matplotlib.pyplot as plt
from cycler import cycler

custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#7f7f7f', '#9467bd', '#e377c2', '#bcbd22', '#17becf'])
plt.rcParams['axes.prop_cycle'] = custom_cycler

class KwargsModelError(Exception):
    '''Custom exception raised for specific error scenarios.'''
    def __init__(self, message):
            super().__init__(message) # Call the base class constructor

class QuasarSpectrum():
    '''Docstring TBD.
    *data: list with [rest wavelength array, flux values array, flux uncertainties array] (all need to have the same shape
    *kwargs_model: dictionary of the form {'continuum': 'powerlaw' or ('polynomial', degree), 
                                            'narrow_doublets': list of pairs (name, nb), where 1+nb is the number of Gauss-Hermite polynomials used
                                            'single_lines': list of pairs (name or wavelength, nb, nature), where 1+nb is the number of Gauss-Hermite polynomials used and nature = 'broad'/'narrow' to use a broad/narrow line prior.
                                            'Template_lines': list of names (only FeII available for now)} '''

    def __init__(self, data, kwargs_model):
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
                self.feature_dict['continuum'] = spec_feat.PolynomialContinuum(lambda_c=central_lambda, degree=kwargs_model['continuum'][1])
            else:
                raise IndexError
        except IndexError:
            raise KwargsModelError('The continuum needs to be either \'powerlaw\' or (\'polynomial\', degree)')

        ## Define narrow-line doublets
        for doublet in kwargs_model['narrow_doublets']:
            self.feature_dict[doublet[0]+'_narrow_doublet'] = spec_feat.NarrowDoublet(name=doublet[0], degree=doublet[1])

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
                self.feature_dict['FeII_template'] = spec_feat.FeIITemplateLines()
            else:
                raise KwargsModelError('Only FeII is supported in Template_lines for the moment.')


    def simulateSpectrum(self, kwargs_values, plot_ax=None):
        '''Docstring TBD.
        Uses a dictionary of parameter values if provided, or the current values stored in the SingleSpectrumParamHandler otherwise.'''
        sim_spec = np.zeros_like(self.lambda_array)
        for feature in self.feature_dict:    
            sim_feat = self.feature_dict[feature].make_flux(self.lambda_array, **kwargs_values[feature])
            sim_spec += sim_feat
            if plot_ax is not None:
                plot_ax.plot(self.lambda_array, sim_feat, label=feature)

        if plot_ax is not None:
            plot_ax.errorbar(self.lambda_array, self.flux_array, xerr=None, yerr=self.flux_err_array, fmt = '.', label='data', color='k', zorder=0)
            plot_ax.plot(self.lambda_array, sim_spec, label='total', color='#d62728')
            plot_ax.legend()
            plt.xlabel(r'Rest wavelength ($\AA$)')
            plt.ylabel(r'Flux (in input units)' )

        return sim_spec

    def get_priors(self, feature):
        '''Docstring TBD.'''
        return self.feature_dict[feature].priors

    def set_priors(self, feature, priors):
        '''Docstring TBD.'''
        self.feature_dict[feature].set_priors(priors)

    def check_bounds(self, kwargs_values, verbose=False):
        '''Returns True if all the parameters within kwargs_values are within the bounds defined in each spectrum component's prior, and False otherwise.
        verbose=True will print a message indicating which parameter was first found to be outside the bounds, if any.'''
        for feature in self.feature_dict:
            for param_name in self.feature_dict[feature].param_names:
                if param_name in ['coeffs','coeffsHermite']: #multiple parameter values in an array
                    if (np.any(kwargs_values[feature][param_name] < self.feature_dict[feature].priors[param_name][2]) or 
                        np.any(kwargs_values[feature][param_name] > self.feature_dict[feature].priors[param_name][3])):                
                        #check if any element of the array is outside the bounds
                        if verbose:
                            print(feature + '_' + param_name + 'is out of prior bounds: value is ', kwargs_values[feature][param_name], 'but bounds are [',
                                  self.feature_dict[feature].priors[param_name][2], ',', self.feature_dict[feature].priors[param_name][3], ']')
                        return False
                else: #one parameter, single numerical value
                    if (kwargs_values[feature][param_name] < self.feature_dict[feature].priors[param_name][2] or
                        kwargs_values[feature][param_name] > self.feature_dict[feature].priors[param_name][3]):
                        if verbose:
                            print(feature + '_' + param_name + 'is out of prior bounds: value is ', kwargs_values[feature][param_name], 'but bounds are [',
                                  self.feature_dict[feature].priors[param_name][2], ',', self.feature_dict[feature].priors[param_name][3], ']')
                        return False
        return True

    def log_likelihood(self, kwargs_values, check_bounds=True, verbose=False):
        '''TBD.'''
        if check_bounds:
            if not(self.check_bounds(kwargs_values=kwargs_values, verbose=verbose)): #if one parameter is outside of the bounds
                return -np.inf

        sim_spec = self.simulateSpectrum(kwargs_values)
        log_lik = np.sum(-(sim_spec - self.flux_array)**2/self.flux_err_array**2)
        return log_lik
        

######### TBD: MCMC SAMPLER ##########

######### TBD: SEPARATE LINEAR (amp) / NONLINEAR PARAMETERS ##########

######### CHECK THAT ALL FEATURES IN THE SPECTRUM ARE >=0 ##########

######### ALLOW PARALLELIZATION ???? ##########

######### LOAD RESULTS FROM A PREVIOUS & POSSIBLY SIMPLER FIT AND SET A PRIOR ##########



        