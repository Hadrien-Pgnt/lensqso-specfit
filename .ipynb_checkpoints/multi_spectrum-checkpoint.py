import numpy as np
import matplotlib.pyplot as plt
import copy

from spectrum_model import QuasarSpectrum

class MultiSpectrum():

    def __init__(self, kwargs_data, kwargs_model, kwargs_fixed):
        '''Docstring TBD.
    *kwargs_data: dictionary with entries of the form {'image_name': [rest wavelength array, flux values array, flux uncertainties array]}
    *kwargs_model: dictionary of the form {'continuum': 'powerlaw' or ('polynomial', degree), 
                                            'narrow_doublets': list of pairs (name, nb), where 1+nb is the number of Gauss-Hermite polynomials used
                                            'single_lines': list of pairs (name or wavelength, nb, nature), where 1+nb is the number of Gauss-Hermite polynomials used and nature = 'broad'/'narrow' to use a broad/narrow line prior.
                                            'Template_lines': list of names (only FeII available for now)} 
    *kwargs_fixed: dictionary of the form {'ref_image':  image_name, 
                                            'fixed_params': dictionary of the form {'feature': list of (non-linear) params fixed to the value of ref image} }'''

        self.spec_dict = {}
        self.narrow_doublet_linHerm = []            
        
        for doublet in kwargs_model['narrow_doublets']:
            key = doublet[0]+'_narrow_doublet'
            if not(key in kwargs_fixed['fixed_params']) or not('relcoeffsHermite' in kwargs_fixed['fixed_params'][key]):
                 #if the Hermite series coefficients are not jointly fit between images, treat them as linear parameters 
                self.narrow_doublet_linHerm.append(doublet[0])

        FeII_linamps = not('FeII_template' in kwargs_fixed['fixed_params']) or np.all([not(key in kwargs_fixed['fixed_params']['FeII_template'])
                                                                                       for key in ['relG', 'relIZw1', 'relS']])
            #if none of the relative amplitude of FeII line families are jointly fit between images, treat them as linear parameters 
    
        for image_name in kwargs_data:
            self.spec_dict[image_name] = QuasarSpectrum(data=kwargs_data[image_name], kwargs_model=kwargs_model, 
                                                        narrow_doublet_linHerm=self.narrow_doublet_linHerm, FeII_linamps = FeII_linamps)

        self.param_handler = ParamHandler(self.spec_dict, **kwargs_fixed)

    def set_initial_values(self):
        '''Updates kwargs_nonlinear_mult in the ParamHandler, using the first number in each prior array as initial values for the spectral features'''
       
        for image_name in self.spec_dict:
            feature_dict = self.spec_dict[image_name].feature_dict
            for feature in feature_dict:
                for param_name in feature_dict[feature].nonlinear_params:
                    if param_name  == 'relcoeffsHermite': #multiple parameter values in an array
                        self.param_handler.kwargs_nonlinear_mult[image_name][feature][param_name][:] = feature_dict[feature].priors[param_name][0]
                    else: #one parameter, single numerical value
                         self.param_handler.kwargs_nonlinear_mult[image_name][feature][param_name] = feature_dict[feature].priors[param_name][0]

        #update the parameters that are fixed to one another
        ref_image = self.param_handler.ref_image
        for feature, fixed_param_list in self.param_handler.fixed_params.items():
            for image_name in self.param_handler.kwargs_nonlinear_mult:
                if not(image_name==ref_image):
                    for param_name in fixed_param_list:
                        if param_name  == 'relcoeffsHermite': #multiple parameter values in an array -> copy all values in that array
                            np.copyto(self.param_handler.kwargs_nonlinear_mult[image_name][feature][param_name], 
                                      self.param_handler.kwargs_nonlinear_mult[ref_image][feature][param_name])
                        else: #single numerical value
                            self.param_handler.kwargs_nonlinear_mult[image_name][feature][param_name] = \
                                self.param_handler.kwargs_nonlinear_mult[ref_image][feature][param_name]

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

    def get_bounds_free_nonlinear_params(self):
        '''Docstring TBD.'''
        N = len(self.param_handler.free_nonlinear_param_list)
        bounds = np.zeros((N,2))
        for i in range(N):
            feature, param_name, image_name = self.param_handler.free_nonlinear_param_list[i].rsplit('_', 2)
            if param_name.startswith('relcoeffsHermite'): 
                bounds[i,:] = self.spec_dict[image_name].feature_dict[feature].priors['relcoeffsHermite'][2:]
            else:
                bounds[i,:] = self.spec_dict[image_name].feature_dict[feature].priors[param_name][2:]
        return bounds  
        
    def get_init_sample_distrib_free_nonlinear_params(self):
        '''Docstring TBD.'''
        N = len(self.param_handler.free_nonlinear_param_list)
        values, sigs = np.zeros(N), np.zeros(N)
        for i in range(N):
            feature, param_name, image_name = self.param_handler.free_nonlinear_param_list[i].rsplit('_', 2)
            if param_name.startswith('relcoeffsHermite'): 
                values[i] = self.spec_dict[image_name].feature_dict[feature].priors['relcoeffsHermite'][0]
                sigs[i] = self.spec_dict[image_name].feature_dict[feature].priors['relcoeffsHermite'][1]
            else:
                values[i] = self.spec_dict[image_name].feature_dict[feature].priors[param_name][0]
                sigs[i] = self.spec_dict[image_name].feature_dict[feature].priors[param_name][1]
        return values, sigs  

    def simulateSpectra(self, kwargs_values_mult, mask_dict={}, tol_positive=-1e-3, plot=False):
        '''Docstring TBD.'''
        sim_specs = {}
        if plot:
            N_img = len(self.spec_dict.keys())
            fig, axes = plt.subplots(N_img, 1, figsize= (10, 5*N_img))
        for i, image_name in enumerate(self.spec_dict):
            mask = mask_dict[image_name] if image_name in mask_dict else np.ones_like(self.spec_dict[image_name].lambda_array)
            sim_specs[image_name], _ = self.spec_dict[image_name].simulateSpectrum(kwargs_values=kwargs_values_mult[image_name], mask_array=mask,
                                                                                   tol_positive=tol_positive, plot_ax = None if not(plot) else axes[i])
            if plot:
                axes[i].set_title('Image ' + image_name)
        return sim_specs

    def check_bounds(self, kwargs_nonlinear_mult=None, verbose=False):
        '''Returns True if all the non-linear parameters within kwargs_nonlinear_mult are within the bounds defined in each spectrum component's prior, and False otherwise.  Uses a dictionary of parameter values if provided, or the current values stored in the ParamHandler otherwise.
        verbose=True will print a message indicating which parameter was first found to be outside the bounds, if any.'''
        if kwargs_nonlinear_mult is None:
            kwargs_nonlinear_mult = self.param_handler.kwargs_nonlinear_mult
        for image_name in self.spec_dict:
            if not(self.spec_dict[image_name].check_bounds(kwargs_nonlinear=kwargs_nonlinear_mult[image_name], verbose=verbose)):
                if verbose:
                    print('this happened for image ' + image_name)
                return False
        return True        

    def log_likelihood_from_kwargs(self, kwargs_nonlinear_mult=None, mask_dict={}, tol_positive=-1e-3, check_bounds=True, verbose=False):
        '''TBD.
        Uses a dictionary of parameter values if provided, or the current values stored in the ParamHandler otherwise.'''
        if check_bounds: #check bounds first to avoid evaluating multiple images if one parameter is outside of the bounds
            if not(self.check_bounds(kwargs_nonlinear_mult=kwargs_nonlinear_mult, verbose=verbose)):
                return -np.inf, None
                
        if kwargs_nonlinear_mult is None:
            kwargs_nonlinear_mult = self.param_handler.kwargs_nonlinear_mult
        
        log_lik = 0
        kwargs_values = {}
        for image_name in self.spec_dict:
            mask = mask_dict[image_name] if image_name in mask_dict else np.ones_like(self.spec_dict[image_name].lambda_array)
            log_lik_image, kwargs_values_image = self.spec_dict[image_name].log_likelihood(kwargs_nonlinear=kwargs_nonlinear_mult[image_name], 
                                                                                           tol_positive=tol_positive, mask_array=mask, check_bounds=False)
            log_lik += log_lik_image
            kwargs_values[image_name] = kwargs_values_image

        return log_lik, kwargs_values

    def log_likelihood_from_array(self, array_values_mult, update=True, **kwargs):
        '''Same as log_likelihood_from_kwargs (in particular, accepts the same kwargs) but the input is an array of free non-linear parameter values (needs to be sorted according to the order in the ParamHandler. Updates the values in the ParamHandler is update is True.'''
        if update:
            self.param_handler.updatekwargs_nonlinear(value_array=array_values_mult)
            return self.log_likelihood_from_kwargs(kwargs_nonlinear_mult=None, **kwargs)
        else:
            kwargs_nonlinear_mult = self.param_handler.array2kwargs_nonlinear(value_array=array_values_mult)
            return self.log_likelihood_from_kwargs(kwargs_nonlinear_mult=kwargs_nonlinear_mult, **kwargs)

class ParamHandler():
    '''For a single MultiSpectrum instance: handles the transformation of arrays of free non-linear parameters (used when sampling) to kwargs dictionaries with all non-linear parameters (used to simulate spectra), and vice-versa. Takes into account the non-linear parameters that are fixed to another parameter's value (specified in kwargs_fixed for MultiSpectrum).'''

    def __init__(self, spec_dict, ref_image, fixed_params):
        self.free_nonlinear_param_list = []
        self.kwargs_nonlinear_mult = {}
        self.fixed_params = fixed_params
        self.ref_image = ref_image

        ### Check if all the parameters that are asked to be fixed in kwargs_fixed are non-linear parameters in the correspoding class, print a warning if not
        for feature in fixed_params:
            for param_fixed in fixed_params[feature]:
                if not(param_fixed in spec_dict[ref_image].feature_dict[feature].nonlinear_params):
                    print('Warning:\'' + param_fixed + '\'  is not part of the non-linear parameters used in class '+ 
                      spec_dict[ref_image].feature_dict[feature].__class__.__name__ +', so this parameter cannot be fixed.' )

        ### Make a list of all the free (non-fixed) non-linear parameters, 
        ### and make a dictionary that has the correct structure with all the parameters non-linear (fixed/free)
        nb_lin_params = 0 # total number of linear parameters
        for image_name in spec_dict:
            kwargs_nonlinear = {}
            nb_lin_params += spec_dict[image_name].lin_param_handler.nb_lin_params
            for feature in spec_dict[image_name].feature_dict:
                kwargs_nonlinear[feature] = {}
                for param_name in spec_dict[image_name].feature_dict[feature].nonlinear_params:
                    is_fixed = (not(image_name==ref_image) and (feature in fixed_params) and (param_name in fixed_params[feature]))
                    if param_name=='relcoeffsHermite': # coefficients of Hermite series for Gauss-Hermite function representation: array of parameters, sorted from lowest to highest order, excluding order 0 when treated as non-linear parameters (relative to order 0)
                        if not(is_fixed):
                            self.free_nonlinear_param_list.extend([feature+'_relcoeffsHermite'+str(i+1)+'_'+image_name 
                                                    for i in range(spec_dict[image_name].feature_dict[feature].degree)])
                        kwargs_nonlinear[feature][param_name] = np.zeros(spec_dict[image_name].feature_dict[feature].degree)
                    else: #single numerical value (NB: polynomial coeffs are necessarily linear so cannot be fixed)
                        if not(is_fixed):
                            self.free_nonlinear_param_list.append(feature+'_'+param_name+'_'+image_name)
                        kwargs_nonlinear[feature][param_name] = 0.

            self.kwargs_nonlinear_mult[image_name] = kwargs_nonlinear

        self.nb_free_params = nb_lin_params + len(self.free_nonlinear_param_list) #total number of free parameters (linear/non-linear)

    
    def updatekwargs_nonlinear(self, value_array):
        '''Method updating the values in *self.kwargs_nonlinear_mult* with an array of free (non-linear) parameter values, respecting the order of parameters, and then update the parameters that are fixed to another.'''
        
        assert (len(value_array) == len(self.free_nonlinear_param_list)) #check that the array has the correct shape
        
        for i in range(len(self.free_nonlinear_param_list)):
            feature, param_name, image_name = self.free_nonlinear_param_list[i].rsplit('_', 2) #free parameters should not be fixed to another
            if param_name.startswith('relcoeffsHermite'): # coefficients of Hermite series sorted from lowest to highest order, excluding order 0 (when treated as non-linear parameters)
                coeff_nb = int(param_name.removeprefix('relcoeffsHermite'))
                self.kwargs_nonlinear_mult[image_name][feature]['relcoeffsHermite'][coeff_nb-1] = value_array[i]
            else:
                self.kwargs_nonlinear_mult[image_name][feature][param_name] = value_array[i]

        #update the parameters that are fixed to one another
        ref_image = self.ref_image
        for feature, fixed_param_list in self.fixed_params.items():
            for image_name in self.kwargs_nonlinear_mult:
                if not(image_name==ref_image):
                    for param_name in fixed_param_list:
                        if param_name == 'relcoeffsHermite': #multiple parameter values in an array -> copy all values in that array
                            np.copyto(self.kwargs_nonlinear_mult[image_name][feature][param_name], self.kwargs_nonlinear_mult[ref_image][feature][param_name])
                        else: #single numerical value (NB: polynomial coeffs are necessarily linear so cannot be fixed)
                            self.kwargs_nonlinear_mult[image_name][feature][param_name] = self.kwargs_nonlinear_mult[ref_image][feature][param_name]

    
    def array2kwargs_nonlinear(self, value_array):
        '''Function returning a NEW kwargs_nonlinear_mult dictionary from an array of free (non-linear) parameter values, respecting the order of parameters, and taking into account the parameters that are fixed to another.'''
        
        assert (len(value_array) == len(self.free_nonlinear_param_list)) #check that the array has the correct shape
        kwargs_nonlinear_mult_new = copy.deepcopy(self.kwargs_nonlinear_mult) #create a deepcopy of the dictionary with the correct structure
        
        for i in range(len(self.free_nonlinear_param_list)):
            feature, param_name, image_name = self.free_nonlinear_param_list[i].rsplit('_', 2) #free parameters should not be fixed to another
            if param_name.startswith('relcoeffsHermite'): #(relative) coefficients of Hermite series sorted from lowest to highest order, excluding order 0 (when treated as non-linear parameters)
                coeff_nb = int(param_name.removeprefix('relcoeffsHermite'))
                kwargs_nonlinear_mult_new[image_name][feature]['relcoeffsHermite'][coeff_nb-1] = value_array[i]
            else:
                kwargs_nonlinear_mult_new[image_name][feature][param_name] = value_array[i]

        #update the parameters that are fixed to one another
        ref_image = self.ref_image
        for feature, fixed_param_list in self.fixed_params.items():
            for image_name in kwargs_nonlinear_mult_new:
                if not(image_name==ref_image):
                    for param_name in fixed_param_list:
                        if param_name == 'relcoeffsHermite': #multiple parameter values in an array -> copy all values in that array
                            np.copyto(kwargs_nonlinear_mult_new[image_name][feature][param_name], kwargs_nonlinear_mult_new[ref_image][feature][param_name])
                        else: #single numerical value (NB: polynomial coeffs are necessarily linear so cannot be fixed)
                            kwargs_nonlinear_mult_new[image_name][feature][param_name] = kwargs_nonlinear_mult_new[ref_image][feature][param_name]

        return kwargs_nonlinear_mult_new
                
    def kwargs_nonlinear2array(self, kwargs_nonlinear_mult=None):
        '''Transforms an dictionary of parameter values into a array, respecting the order of parameters and only considering the free parameters.'''
        if kwargs_nonlinear_mult is None:
            kwargs_nonlinear_mult = self.kwargs_nonlinear_mult
        n_params = len(self.free_nonlinear_param_list)
        array_values = np.zeros(n_params)
        for i in range(n_params):
            feature, param_name, image_name = self.free_nonlinear_param_list[i].rsplit('_', 2)
            if param_name.startswith('relcoeffsHermite'): #(relative) coefficients of Hermite series sorted from lowest to highest order, excluding order 0 (when treated as non-linear parameters)
                coeff_nb = int(param_name.removeprefix('relcoeffsHermite'))
                array_values[i] = kwargs_nonlinear_mult[image_name][feature]['relcoeffsHermite'][coeff_nb-1]
            else: #single numerical value (NB: polynomial coeffs are necessarily linear so cannot be fixed)
                array_values[i] = kwargs_nonlinear_mult[image_name][feature][param_name]

        return array_values




