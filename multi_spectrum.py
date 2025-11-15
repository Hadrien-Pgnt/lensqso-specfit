import numpy as np
import matplotlib.pyplot as plt
from spectrum_model import QuasarSpectrum
import copy

class MultiSpectrum():

    def __init__(self, kwargs_data, kwargs_model, kwargs_fixed={}):
        '''Docstring TBD.
    *kwargs_data: dictionary with entries of the form {'image_name': [rest wavelength array, flux values array, flux uncertainties array]}
    *kwargs_model: dictionary of the form {'continuum': 'powerlaw' or ('polynomial', degree), 
                                            'narrow_doublets': list of pairs (name, nb), where 1+nb is the number of Gauss-Hermite polynomials used
                                            'single_lines': list of pairs (name or wavelength, nb, nature), where 1+nb is the number of Gauss-Hermite polynomials used and nature = 'broad'/'narrow' to use a broad/narrow line prior.
                                            'Template_lines': list of names (only FeII available for now)} 
    *kwargs_fixed: dictionary of the form {'ref_image':  image_name, 
                                            'fixed_params': dictionary of the form {'feature': list of params fixed to the value of ref image} }'''

        self.spec_dict = {}
        for image_name in kwargs_data:
            self.spec_dict[image_name] = QuasarSpectrum(data=kwargs_data[image_name], kwargs_model=kwargs_model)

        self.param_handler = ParamHandler(self.spec_dict, kwargs_fixed=kwargs_fixed)

    def set_initial_values(self):
        '''Updates the values in the ParamHandler, using the first number in each prior array as initial values for the spectral features'''
       
        for image_name in self.spec_dict:
            feature_dict = self.spec_dict[image_name].feature_dict
            for feature in feature_dict:
                for param_name in feature_dict[feature].param_names:
                    if param_name in ['coeffs','coeffsHermite']: #multiple parameter values in an array
                        self.param_handler.kwargs_values_mult[image_name][feature][param_name][:] = feature_dict[feature].priors[param_name][0]
                    else: #one parameter, single numerical value
                         self.param_handler.kwargs_values_mult[image_name][feature][param_name] = feature_dict[feature].priors[param_name][0]

        #update the parameters that are fixed to one another
        ref_image = self.param_handler.kwargs_fixed['ref_image']
        for feature, fixed_param_list in self.param_handler.kwargs_fixed['fixed_params'].items():
            for image_name in self.param_handler.kwargs_values_mult:
                if not(image_name==ref_image):
                    for param_name in fixed_param_list:
                        if param_name in ['coeffs','coeffsHermite']: #multiple parameter values in an array -> copy all values in that array
                            np.copyto(self.param_handler.kwargs_values_mult[image_name][feature][param_name], 
                                      self.param_handler.kwargs_values_mult[ref_image][feature][param_name])
                        else: #single numerical value
                            self.param_handler.kwargs_values_mult[image_name][feature][param_name] = \
                                self.param_handler.kwargs_values_mult[ref_image][feature][param_name]

    def get_priors(self, feature, images='all'):
        '''Docstring TBD.'''
        prior_dict = {}
        if images == 'all':
            images = self.spec_dict.keys()
        for image_name in images:
            prior_dict[image_name] = self.spec_dict[image_name].get_priors(feature)
        return prior_dict

    def set_priors(self, feature, priors, images='all'):
        '''Docstring TBD.'''
        if images == 'all':
            images = self.spec_dict.keys()
        for image_name in images:
            self.spec_dict[image_name].set_priors(feature, priors)

    def get_bounds_freeparams(self):
        '''Docstring TBD.'''
        N = len(self.param_handler.free_param_list)
        bounds = np.zeros((N,2))
        for i in range(N):
            feature, param_name, image_name = self.param_handler.free_param_list[i].rsplit('_', 2)
            if param_name.startswith('coeffsHermite'): 
                bounds[i,:] = self.spec_dict[image_name].feature_dict[feature].priors['coeffsHermite'][2:]
            elif param_name.startswith('coeffs'): 
                bounds[i,:] = self.spec_dict[image_name].feature_dict[feature].priors['coeffs'][2:]
            else:
                bounds[i,:] = self.spec_dict[image_name].feature_dict[feature].priors[param_name][2:]
        return bounds  
        
    def get_init_sample_distrib(self):
        N = len(self.param_handler.free_param_list)
        values, sigs = np.zeros(N), np.zeros(N)
        for i in range(N):
            feature, param_name, image_name = self.param_handler.free_param_list[i].rsplit('_', 2)
            if param_name.startswith('coeffsHermite'): 
                values[i] = self.spec_dict[image_name].feature_dict[feature].priors['coeffsHermite'][0]
                sigs[i] = self.spec_dict[image_name].feature_dict[feature].priors['coeffsHermite'][1]
            elif param_name.startswith('coeffs'): 
                values[i] = self.spec_dict[image_name].feature_dict[feature].priors['coeffs'][0]
                sigs[i] = self.spec_dict[image_name].feature_dict[feature].priors['coeffs'][1]
            else:
                values[i] = self.spec_dict[image_name].feature_dict[feature].priors[param_name][0]
                sigs[i] = self.spec_dict[image_name].feature_dict[feature].priors[param_name][1]
        return values, sigs  

    def simulateSpectra(self, kwargs_values_mult=None, plot=False):
        '''Docstring TBD.'''
        sim_specs = {}
        if kwargs_values_mult is None:
            kwargs_values_mult = self.param_handler.kwargs_values_mult
        if plot:
            N_img = len(self.spec_dict.keys())
            fig, axes = plt.subplots(N_img, 1, figsize= (10, 5*N_img))
        for i, image_name in enumerate(self.spec_dict):
            sim_specs[image_name] = self.spec_dict[image_name].simulateSpectrum(kwargs_values=kwargs_values_mult[image_name], 
                                                                                plot_ax = None if not(plot) else axes[i])
            if plot:
                axes[i].set_title('Image ' + image_name)
        return sim_specs

    def check_bounds(self, kwargs_values_mult=None, verbose=False):
        '''Returns True if all the parameters within kwargs_values_mult are within the bounds defined in each spectrum component's prior, and False otherwise.
        Uses a dictionary of parameter values if provided, or the current values stored in the ParamHandler otherwise.
        verbose=True will print a message indicating which parameter was first found to be outside the bounds, if any.'''
        if kwargs_values_mult is None:
            kwargs_values_mult = self.param_handler.kwargs_values_mult
        for image_name in self.spec_dict:
            if not(self.spec_dict[image_name].check_bounds(kwargs_values=kwargs_values_mult[image_name], verbose=verbose)):
                if verbose:
                    print('this happened for image ' + image_name)
                return False
        return True        

    def log_likelihood_from_kwargs(self, kwargs_values_mult=None, check_bounds=True, verbose=False):
        '''TBD.
        Uses a dictionary of parameter values if provided, or the current values stored in the ParamHandler otherwise.'''
        if check_bounds: #check bounds first to avoid evaluating multiple images if one parameter is outside of the bounds
            if not(self.check_bounds(kwargs_values_mult=kwargs_values_mult, verbose=verbose)):
                return -np.inf
                
        if kwargs_values_mult is None:
            kwargs_values_mult = self.param_handler.kwargs_values_mult
        
        log_lik = 0
        for image_name in self.spec_dict:
            log_lik += self.spec_dict[image_name].log_likelihood(kwargs_values=kwargs_values_mult[image_name], check_bounds=False)
        return log_lik

    def log_likelihood_from_array(self, array_values_mult, update=True, check_bounds=True, verbose=False):
        '''TBD.
        Updates the values in the ParamHandler is update is True.'''
        if update:
            self.param_handler.updatekwargs(value_array=array_values_mult)
            return self.log_likelihood_from_kwargs(kwargs_values_mult=None, check_bounds=check_bounds, verbose=verbose)
        else:
            kwargs_values_mult = self.param_handler.array2kwargs(value_array=array_values_mult)
            return self.log_likelihood_from_kwargs(kwargs_values_mult=kwargs_values_mult, check_bounds=check_bounds, verbose=verbose)

class ParamHandler():
    '''For a single MultiSpectrum instance:
    handles the transformation of arrays (used when sampling) to kwargs dictionaries (use to simulate spectra), and vice-versa.
    Takes into account the parameters that are fixed to another parameter's value (specified in kwargs_fixed) '''

    def __init__(self, spec_dict, kwargs_fixed):
        self.free_param_list = []
        self.kwargs_values_mult = {}
        self.kwargs_fixed = kwargs_fixed

        for image_name in spec_dict:
            kwargs_values = {}
            for feature in spec_dict[image_name].feature_dict:
                kwargs_values[feature] = {}
                for param_name in spec_dict[image_name].feature_dict[feature].param_names:
                    is_fixed = (not(image_name==self.kwargs_fixed['ref_image']) and feature in self.kwargs_fixed['fixed_params'] 
                                and param_name in self.kwargs_fixed['fixed_params'][feature])
                    if param_name=='coeffsHermite': #coefficients of Hermite series for Gauss-Hermite function representation: array of parameters, sorted from lowest to highest order, excluding order 0
                        if not(is_fixed):
                            self.free_param_list.extend([feature+'_'+param_name+str(i+1)+'_'+image_name 
                                                    for i in range(spec_dict[image_name].feature_dict[feature].degree)])
                        kwargs_values[feature][param_name] = np.zeros(spec_dict[image_name].feature_dict[feature].degree)
                    elif param_name=='coeffs': #polynomial coefficients: array of parameters, sorted from highest to lowest order, excluding order 0
                        N = spec_dict[image_name].feature_dict[feature].degree
                        if not(is_fixed):
                            self.free_param_list.extend([feature+'_'+param_name+str(N-i)+'_'+image_name for i in range(N)])
                        kwargs_values[feature][param_name] = np.zeros(N)
                    else: #single numerical value
                        if not(is_fixed):
                            self.free_param_list.append(feature+'_'+param_name+'_'+image_name)
                        kwargs_values[feature][param_name] = 0.

            self.kwargs_values_mult[image_name] = kwargs_values
            
    def updatekwargs(self, value_array):
        '''Method updating the values in *self.kwargs_values_mult* with an array of (free) parameter values, respecting the order of parameters, and then update the parameters that are fixed to another.'''
        
        assert (len(value_array) == len(self.free_param_list)) #check that the array has the correct shape
        
        for i in range(len(self.free_param_list)):
            feature, param_name, image_name = self.free_param_list[i].rsplit('_', 2) #free parameters should not be fixed to another
            if param_name.startswith('coeffsHermite'): #coefficients of Hermite series are sorted from lowest to highest order, excluding order 0
                coeff_nb = int(param_name.removeprefix('coeffsHermite'))
                self.kwargs_values_mult[image_name][feature]['coeffsHermite'][coeff_nb-1] = value_array[i]
            elif param_name.startswith('coeffs'): #polynomial coefficients are sorted from highest to lowest order, excluding order 0
                coeff_nb = int(param_name.removeprefix('coeffs'))
                N = len(self.kwargs_values_mult[image_name][feature]['coeffs']) 
                self.kwargs_values_mult[image_name][feature]['coeffs'][N-coeff_nb] = value_array[i]
            else:
                self.kwargs_values_mult[image_name][feature][param_name] = value_array[i]

        #update the parameters that are fixed to one another
        ref_image = self.kwargs_fixed['ref_image']
        for feature, fixed_param_list in self.kwargs_fixed['fixed_params'].items():
            for image_name in self.kwargs_values_mult:
                if not(image_name==ref_image):
                    for param_name in fixed_param_list:
                        if param_name in ['coeffs','coeffsHermite']: #multiple parameter values in an array -> copy all values in that array
                            np.copyto(self.kwargs_values_mult[image_name][feature][param_name], self.kwargs_values_mult[ref_image][feature][param_name])
                        else: #single numerical value
                            self.kwargs_values_mult[image_name][feature][param_name] = self.kwargs_values_mult[ref_image][feature][param_name]
                            

    def array2kwargs(self, value_array):
        '''Function returning a NEW kwargs_values_mult dictionary from an array of parameter values, respecting the order of parameters'''
        
        assert (len(value_array) == (len(self.free_param_list),)) #check that the array has the correct shape
        kwargs_values_mult_new = copy.deepcopy(self.kwargs_values_mult) #create a deepcopy of the dictionary with the correct structure
        
        for i in range(len(self.free_param_list)):
            feature, param_name, image_name = self.free_param_list[i].rsplit('_', 2)
            if param_name.startswith('coeffsHermite'): #coefficients of Hermite series are sorted from lowest to highest order, excluding order 0
                coeff_nb = int(param_name.removeprefix('coeffsHermite'))
                kwargs_values_mult_new[image_name][feature]['coeffsHermite'][coeff_nb-1] = value_array[i]
            elif param_name.startswith('coeffs'): #polynomial coefficients are sorted from highest to lowest order, excluding order 0
                coeff_nb = int(param_name.removeprefix('coeffs'))
                N = len(kwargs_values_mult_new[image_name][feature]['coeffs']) 
                kwargs_values_mult_new[image_name][feature]['coeffs'][N-coeff_nb] = value_array[i]
            else:
                kwargs_values_mult_new[image_name][feature][param_name] = value_array[i]

        #assign the correct values to the parameters that are fixed to one another
        ref_image = self.kwargs_fixed['ref_image']
        for feature, fixed_param_list in self.kwargs_fixed['fixed_params'].items():
            for image_name in kwargs_values_mult_new:
                if not(image_name==ref_image):
                    for param_name in fixed_param_list:
                        if param_name in ['coeffs','coeffsHermite']: #multiple parameter values in an array -> copy all values in that array
                            np.copyto(kwargs_values_mult_new[image_name][feature][param_name], kwargs_values_mult_new[ref_image][feature][param_name])
                        else: #single numerical value
                            kwargs_values_mult_new[image_name][feature][param_name] = kwargs_values_mult_new[ref_image][feature][param_name]

        return kwargs_values_mult_new

    def kwargs2array(self, kwargs_values_mult=None):
        '''Transforms an dictionary of parameter values into a array, respecting the order of parameters and only considering the free parameters.'''
        if kwargs_values_mult is None:
            kwargs_values_mult = self.kwargs_values_mult
        n_params = len(self.free_param_list)
        array_values = np.zeros(n_params)
        for i in range(n_params):
            feature, param_name, image_name = self.free_param_list[i].rsplit('_', 2)
            if param_name.startswith('coeffsHermite'): #coefficients of Hermite series are sorted from lowest to highest order, excluding order 0
                coeff_nb = int(param_name.removeprefix('coeffsHermite'))
                array_values[i] = kwargs_values_mult[image_name][feature]['coeffsHermite'][coeff_nb-1]
            elif param_name.startswith('coeffs'): #polynomial coefficients are sorted from highest to lowest order, excluding order 0
                coeff_nb = int(param_name.removeprefix('coeffs'))
                N = len(kwargs_values_mult[image_name][feature]['coeffs']) 
                array_values[i] = kwargs_values_mult[image_name][feature]['coeffs'][N-coeff_nb] 
            else:
                array_values[i] = kwargs_values_mult[image_name][feature][param_name]

        return array_values






