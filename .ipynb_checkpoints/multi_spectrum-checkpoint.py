import numpy as np
import matplotlib.pyplot as plt
import copy

from spectrum_model import QuasarSpectrum

class MultiSpectrum():

    def __init__(self, kwargs_data, kwargs_model, kwargs_fixed):
        '''Class to jointly model multiple quasar spectra with the same spectral features (continuum + emission lines, separated between singlets, narrow doublets, and template of families of lines), e.g. the spectra from the different images of a multiply imaged quasar.
        
    *kwargs_data: dictionary with entries of the form {'image_name': [rest wavelength array, flux values array, flux uncertainties array]}
    
    *kwargs_model: dictionary of the form {
        'continuum': 'powerlaw' or ('polynomial', degree), 
        'narrow_doublets': list of (name, type, kwargs_init), 
                where type ='GaussHermite'/'Voigt' 
                and kwargs_init={'degree':int, 'fix_c1c2_to_zero':bool} for Gauss-Hermite profiles (and ={} for Voigt).
                                            
        'single_lines': list of (name or wavelength, type, kwargs_init, nature), 
                where type ='GaussHermite'/'Voigt', 
                kwargs_init={'degree':int, 'fix_c1c2_to_zero':bool} for Gauss-Hermite profiles (and ={} for Voigt),
                and nature = 'broad'/'narrow' to use adequate priors. prior.
        'Template_lines': list of names (only 'FeII_Vis' and 'FeII+MgII_NIR' available for now)
        }.
        
    *kwargs_fixed: dictionary of the form {'ref_image':  image_name, 
                                            'fixed_params': dictionary of the form {'feature': list of (non-linear) params} }.
                    If a parameter is in fixed_params, its value will be shared across all images (and only the value for ref_image will be sampled).'''

        self.spec_dict = {}
        self.lines_relHerm = []            
        
        for doublet in kwargs_model['narrow_doublets']:
            key = doublet[0]+'_narrow_doublet'
            if (key in kwargs_fixed['fixed_params']) and ('relcoeffsHermite' in kwargs_fixed['fixed_params'][key]):
                 #if the Hermite series coefficients are jointly fit between images, treat the order >=1 ones as non-linear parameters 
                self.lines_relHerm.append(doublet[0])
        for line in kwargs_model['single_lines']:
            key = line[0]+'_'+line[3]
            if (key in kwargs_fixed['fixed_params']) and ('relcoeffsHermite' in kwargs_fixed['fixed_params'][key]):
                 #if the Hermite series coefficients are jointly fit between images, treat the order >=1 ones as non-linear parameters 
                self.lines_relHerm.append(line[0])

        FeII_Vis_linamps = not('FeII_Vis_template' in kwargs_fixed['fixed_params']) or np.all([not(key in kwargs_fixed['fixed_params']['FeII_Vis_template'])
                                                                                       for key in ['relG', 'relIZw1', 'relS']])
            #if none of the relative amplitude of FeII line families are jointly fit between images, treat them as linear parameters 
    
        for image_name in kwargs_data:
            self.spec_dict[image_name] = QuasarSpectrum(data=kwargs_data[image_name], kwargs_model=kwargs_model, 
                                                        lines_relHerm=self.lines_relHerm, FeII_Vis_linamps = FeII_Vis_linamps)

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
        '''Returns the priors on the parameters describing the spectral feature *feature*.
        Parameter *images* takes a list of image names and returns the priors for each of those images, if ='all', returns for all images.'''
        prior_dict = {}
        if images == 'all':
            images = self.spec_dict.keys()
        for image_name in images:
            prior_dict[image_name] = self.spec_dict[image_name].get_priors(feature)
        return prior_dict

    def set_priors(self, feature, priors, images='all'):
        '''Updates the priors on the parameters describing the spectral feature *feature*, with the values contained in *priors*.
        Parameter *images* takes a list of image names and returns the priors for each of those images, if ='all', returns for all images.'''
        if images == 'all':
            images = self.spec_dict.keys()
        for image_name in images:
            self.spec_dict[image_name].set_priors(feature, priors)

    def get_bounds_free_nonlinear_params(self):
        '''Returns an array with the upper/lower bounds in the priors for all of the free (i.e., not fixed to another value) and non-linear parameters.
        The parameters are sorted according to the order in the ParamHandler.'''
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
        '''Returns 2 arrays with the means and std deviations characterizing the Gaussian priors for all of the free and non-linear parameters.
        The parameters are sorted according to the order in the ParamHandler.'''
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


    def get_fluxes_of_feature(self, kwargs_values_mult, feature):
        '''For each image, returns the total flux of the same single line or doublet.'''

        if feature.endswith('_template'):
            raise CustomError('The total flux in template lines depends on the wavelength range !')
        elif feature=='continuum':
            raise CustomError('The total continuum flux depends on the wavelength range !')
        else:
            fluxes = {}
            for image_name in self.spec_dict: 
                fluxes[image_name] = self.spec_dict[image_name].get_flux_of_feature(kwargs_values_mult[image_name], feature)
            return fluxes

    def get_flux_ratios_of_feature(self, kwargs_values_mult, feature, ref_image=None):
        '''For each image, returns the flux ratios relative to *ref_image* for a single line or a doublet.
        If ref_image is None, return all the possible flux ratios.'''
        
        fluxes = self.get_fluxes_of_feature(kwargs_values_mult, feature)
        flux_ratios = {}
        if ref_image is not None:
            assert (ref_image in self.spec_dict.keys())
            for image_name in self.spec_dict: 
                if not(image_name==ref_image):
                    flux_ratios['f_'+image_name+'/f_'+ref_image] = fluxes[image_name]/fluxes[ref_image]
        else:
            for image1 in self.spec_dict: 
                for image2 in self.spec_dict:
                    if not(image1==image2):
                        flux_ratios['f_'+image1+'/f_'+image2] = fluxes[image1]/fluxes[image2]
            
        return flux_ratios
        

    def simulateSpectra(self, kwargs_values_mult, mask_dict={}, tol_positive=-1e-3, tol_positive_dict={}, 
                        plot=False, norm_residuals=False, print_res_stats=False):
        '''Given a set of parameter values, generates the model spectra for all images by calculating the flux for each wavelength in self.lambda_array.
        
        Inputs:
        *kwargs_values_mult:  dictionary of the form {image1: kwargs_values1, image2: kwargs_values2,...} where kwargs_values contains numerical values for all the parameters (linear AND nonlinear) of the corresponding image.
        *mask_dict: dictionary of the form {image1: mask1, image2: mask2,...} where mask is boolean array indicating which wavelengths should be included (in the likelihood calculation / plot / optimization of linear parameters) for the corresponding image.
        *tol_positive: consider that fluxes above this value are still positive. Use this values for all the images unless specified in tol_positive_dict.
        *tol_positive_dict: dictionary of the form {image1: tol1, image2: tol2,...} where tol is a value above which fluxes are still considered positive (numerical tolerance) for the corresponding image.
        *plot: boolean. If True, will plot the fits.
        *norm_residuals: If True, will plot normalized resiudals (otherwise absolute residuals with errorbars) for each image.
        *print_res_stats: if True, print the mean and std deviation of the normalized residuals for each image.
        
        Outputs:
        * sim_specs:  dictionary of the form {image1: sim_spec1, image2: sim_spec2,...} where sim_spec is the simulated spectrum for the corresponding image.'''
        sim_specs = {}
        if plot:
            N_img = len(self.spec_dict.keys())
            #fig, axes = plt.subplots(2*N_img, 1, figsize= (10, 7*N_img), height_ratios=[5,2]*N_img)
            axes_nested = []
            fig, axes = plt.subplots(N_img, 1, layout='constrained', figsize= (10, 7*N_img))
            gridspec = axes[0].get_subplotspec().get_gridspec()
            for i in range(N_img):
                axes[i].remove()
                subfig = fig.add_subfigure(gridspec[i])
                axes_i = subfig.subplots(2, 1, sharex=True, height_ratios=[5,2])
                axes_nested.append(axes_i)

        for i, image_name in enumerate(self.spec_dict):
            mask = mask_dict[image_name] if image_name in mask_dict else np.ones_like(self.spec_dict[image_name].lambda_array)
            tol_pos = tol_positive_dict[image_name] if image_name in tol_positive_dict else tol_positive
            if print_res_stats:
                print('For image ' + image_name + ':')
            sim_specs[image_name], _ = self.spec_dict[image_name].simulateSpectrum(kwargs_values=kwargs_values_mult[image_name], 
                                                                                   mask_array=mask, tol_positive=tol_pos, 
                                                                                   plot_axes = None if not(plot) else axes_nested[i],
                                                                                   norm_residuals=norm_residuals, print_res_stats=print_res_stats)
            if plot:
                axes_nested[i][0].get_figure().suptitle('Image ' + image_name)
        
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

    def log_likelihood_from_kwargs(self, kwargs_nonlinear_mult=None, mask_dict={}, tol_positive=-1e-3, tol_positive_dict={}, check_bounds=True, verbose=False):
        '''Uses the input spectra given during the initialization (flux with uncertainties for each wavelength) to compute the log-likelihood (joint across the multiple images) for a given set of non-linear parameters (the linear parameters are automatically optimized, using a MLE estimate).
        
        Inputs:
        *kwargs_nonlinear_mult:  dictionary of the form {image1: kwargs_nonlin1, image2: kwargs_nonlin2,...} where kwargs_nonlin contains numerical values for all the non-linear parameters of the corresponding image. If =None, the values stored in the ParamHandler are used.
        *mask_dict: dictionary of the form {image1: mask1, image2: mask2,...} where mask is boolean array indicating which wavelengths should be included for the corresponding image.
        *tol_positive: consider that fluxes above this value are still positive. Use this values for all the images unless specified in tol_positive_dict.
        *tol_positive_dict: dictionary of the form {image1: tol1, image2: tol2,...} where tol is a value above which fluxes are still considered positive (numerical tolerance) for the corresponding image.
        *check_bounds: if True, will check if all the non-linear parameters are within the bounds of their prior, and set the log-likelihood to -inf if one of them is outside the bounds.
        *verbose: if True, will print which parameter is outside the bounds and for which image, if any.
        
        Outputs:
        *log_lik: the log-likelihood value
        *kwargs_values: a dictionary with all the parameter values (linear AND non-linear)
        '''
        if check_bounds: #check bounds first to avoid evaluating multiple images if one parameter is outside of the bounds
            if not(self.check_bounds(kwargs_nonlinear_mult=kwargs_nonlinear_mult, verbose=verbose)):
                return -np.inf, None
                
        if kwargs_nonlinear_mult is None:
            kwargs_nonlinear_mult = self.param_handler.kwargs_nonlinear_mult
        
        log_lik = 0
        kwargs_values = {}
        for image_name in self.spec_dict:
            mask = mask_dict[image_name] if image_name in mask_dict else np.ones_like(self.spec_dict[image_name].lambda_array)
            tol_pos = tol_positive_dict[image_name] if image_name in tol_positive_dict else tol_positive
            log_lik_image, kwargs_values_image = self.spec_dict[image_name].log_likelihood(kwargs_nonlinear=kwargs_nonlinear_mult[image_name], 
                                                                                           tol_positive=tol_pos, mask_array=mask, check_bounds=False)
            log_lik += log_lik_image
            kwargs_values[image_name] = kwargs_values_image

        return log_lik, kwargs_values

    def log_likelihood_from_array(self, array_values_mult, update=True, **kwargs):
        '''Same as log_likelihood_from_kwargs (in particular, accepts the same kwargs) but instead of a dictionary (kwargs_nonlinear_mult), the main input is an array of free non-linear parameter values *array_values_mult* (needs to be sorted according to the order in the ParamHandler). 
        Updates the values in the ParamHandler if *update* is True.'''
        if update:
            self.param_handler.updatekwargs_nonlinear(value_array=array_values_mult)
            return self.log_likelihood_from_kwargs(kwargs_nonlinear_mult=None, **kwargs)
        else:
            kwargs_nonlinear_mult = self.param_handler.array2kwargs_nonlinear(value_array=array_values_mult)
            return self.log_likelihood_from_kwargs(kwargs_nonlinear_mult=kwargs_nonlinear_mult, **kwargs)

class ParamHandler():
    '''Class to facilitate the handling of linear parameters in MultiSpectrum: 
transformation of arrays of free non-linear parameters (used when sampling) to kwargs dictionaries with all non-linear parameters (used to simulate spectra), and vice-versa. Takes into account the non-linear parameters that are fixed to another parameter's value (specified in kwargs_fixed for MultiSpectrum).'''

    def __init__(self, spec_dict, ref_image, fixed_params):
        self.free_nonlinear_param_list = []
        self.kwargs_nonlinear_mult = {}
        self.fixed_params = fixed_params
        self.fixedc1c2_in_GH = [] #list of features where c1=c2=0 will be fixed in the Gauss-Hermite series
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
                    if param_name=='relcoeffsHermite': # coefficients of Hermite series for Gauss-Hermite function representation: array of parameters, sorted from lowest to highest order, excluding order 0 when treated as non-linear parameters (relative to order 0), and excluding orders 1 and 2 if specified.
                        fixc1c2 = spec_dict[image_name].feature_dict[feature].fix_c1c2_to_zero
                        if fixc1c2:
                            self.fixedc1c2_in_GH.append(feature) 
                        if not(is_fixed):
                            self.free_nonlinear_param_list.extend([feature+'_relcoeffsHermite'+str(i+1)+'_'+image_name 
                                                    for i in range(2*fixc1c2, spec_dict[image_name].feature_dict[feature].degree)])
                        kwargs_nonlinear[feature][param_name] = np.zeros(max(0,spec_dict[image_name].feature_dict[feature].degree - 2*fixc1c2))
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
            if param_name.startswith('relcoeffsHermite'): # coefficients of Hermite series sorted from lowest to highest order, excluding order 0 (when treated as non-linear parameters), and excluding orders 1 and 2 if specified.
                coeff_nb = int(param_name.removeprefix('relcoeffsHermite'))
                fixc1c2 = (feature in self.fixedc1c2_in_GH)
                self.kwargs_nonlinear_mult[image_name][feature]['relcoeffsHermite'][coeff_nb-1-2*fixc1c2] = value_array[i]
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
            if param_name.startswith('relcoeffsHermite'): #(relative) coefficients of Hermite series sorted from lowest to highest order, excluding order 0 (when treated as non-linear parameters), and excluding orders 1 and 2 if specified.
                coeff_nb = int(param_name.removeprefix('relcoeffsHermite'))
                fixc1c2 = (feature in self.fixedc1c2_in_GH)
                kwargs_nonlinear_mult_new[image_name][feature]['relcoeffsHermite'][coeff_nb-1-2*fixc1c2] = value_array[i]
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
            if param_name.startswith('relcoeffsHermite'): #(relative) coefficients of Hermite series sorted from lowest to highest order, excluding order 0 (when treated as non-linear parameters), and excluding orders 1 and 2 if specified.
                coeff_nb = int(param_name.removeprefix('relcoeffsHermite'))
                fixc1c2 = (feature in self.fixedc1c2_in_GH)
                array_values[i] = kwargs_nonlinear_mult[image_name][feature]['relcoeffsHermite'][coeff_nb-1-2*fixc1c2]
            else: #single numerical value (NB: polynomial coeffs are necessarily linear so cannot be fixed)
                array_values[i] = kwargs_nonlinear_mult[image_name][feature][param_name]

        return array_values




