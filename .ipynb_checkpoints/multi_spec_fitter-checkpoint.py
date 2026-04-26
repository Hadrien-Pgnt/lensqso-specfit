import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from scipy.optimize import minimize
from scipy import stats
from math import floor
import math
from tqdm import tqdm
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer, Particle
from lenstronomy.Plots.chain_plot import plot_chain
import emcee

from multi_spectrum import MultiSpectrum
from spectrum_model import CustomError


class Optimizer():
    '''Base class to optimize the parameters in a joint model of multiple quasar spectra with the same spectral features.

        Inputs:
        *multi_spec: instance of MultiSpectrum() class
        *kwargs_likelihood: dictionary of the form {'mask_dict': dictionary of mask arrays for each image, 
                                                    'tol_positive_dict': dictionary of numerical tolerance for flux positivity, for each image}.
        *init_array: array of values for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess during the optimization. If None, the mean values of the priors are used.
        *sigs_array: array of uncertainties for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess for the spread of values during the optimization. If None, the std deviations of the priors are used.
        '''

    def __init__(self, multi_spec, kwargs_likelihood, init_array=None, sigs_array=None):

        self.multi_spec = multi_spec
        
        self.multi_spec.set_initial_values()

        # initial values of free (non-linear) parameters + spread of initial samples
        init_array_default, sig_array_default = self.multi_spec.get_init_sample_distrib_free_nonlinear_params() 
        self.init_array = init_array_default if init_array is None else init_array
        self.sigs_array = sig_array_default if sigs_array is None else sigs_array
                                
        self.bounds_array = self.multi_spec.get_bounds_free_nonlinear_params() #prior bounds on free (non-linear) parameters 
        self.kwargs_lik = kwargs_likelihood
        

    def find_MLE(self, plot=True, return_array=False, print_res_stats=True, norm_residuals=False):
        ''' Uses the COBYQA method for constrained optimization (needs scipy>=1.14.0) to find the values for the free, non-linear parameters maximizing the likelihood. Should update the ParamHandler kwargs values automatically in self.multi_spec.
        Uses the lower/upper bounds prescribed in the priors.
        If plot is True, displays the best-fit model.
        
        Outputs:
        *logL: the maximum log-likelihood value that was found during the optimization
        *kwargs_values: the dictionary of all parameter values (linear and non-linear, free and fixed) maximizing the likelihood.
        If return_array is True, also returns *res.x*, the array of free, non-linear parameters maximizing the likelihood (sorted according to the order in the ParamHandler of self.multi_spec).'''
        res = minimize(lambda a : -self.multi_spec.log_likelihood_from_array(a, **self.kwargs_lik)[0], 
                       x0 = self.init_array, bounds = self.bounds_array, method='COBYQA')
        
        logL, kwargs_values = self.multi_spec.log_likelihood_from_array(res.x, **self.kwargs_lik)
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values, plot=True, print_res_stats=print_res_stats, norm_residuals=norm_residuals,
                                                **self.kwargs_lik)

        if return_array:
            return logL, res.x, kwargs_values
        else:
            return logL, kwargs_values



class PSO(Optimizer):
    '''Child class of Optimizer, using a Point Swarm Optimization to find the max likelihood.
     Inputs:
        *multi_spec: instance of MultiSpectrum() class
        *kwargs_likelihood: dictionary of the form {'mask_dict': dictionary of mask arrays for each image, 
                                                    'tol_positive': float}.
        *n_particles: number of particles used by the PSO.
         *init_array: array of values for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess during the optimization. If None, the mean values of the priors are used.
        *sigs_array: array of uncertainties for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess for the spread of values during the optimization. If None, the std deviations of the priors are used.

    '''
    
    def __init__(self, multi_spec, n_particles, kwargs_likelihood={}, init_array=None, sigs_array=None):
        super().__init__(multi_spec, kwargs_likelihood, init_array, sigs_array)
        self.pso = ParticleSwarmOptimizer(func=(lambda a : self.multi_spec.log_likelihood_from_array(a, **self.kwargs_lik)[0]), particle_count = n_particles,
                                          low=self.bounds_array[:,0], high=self.bounds_array[:,1])

        ## Initialize with samples (override uniform intialization from lenstronomy, with Gaussian samples with mean in init_array and std dev in sigs_array,
        ## truncated such that all the samples are within the bounds)
        swarm = []
        norm_lower_limits = (self.bounds_array[:,0] - self.init_array) / self.sigs_array
        norm_upper_limits = (self.bounds_array[:,1] - self.init_array) / self.sigs_array
        for _ in range(self.pso.particleCount):
            swarm.append(Particle(self.init_array + self.sigs_array * stats.truncnorm.rvs(norm_lower_limits, norm_upper_limits, size=self.pso.param_count),
                                  np.zeros(self.pso.param_count)))
        self.pso.swarm = swarm

    def optimize(self, plot=True, return_all=False, print_res_stats=True, norm_residuals=False, **kwargs):
        '''Uses a PSO algorithm (relies on the implementation in lenstronomy) to find the values for the free, non-linear parameters maximizing the likelihood. Should update the ParamHandler kwargs values automatically in self.multi_spec.
        Uses the lower/upper bounds prescribed in the priors.
        If plot is True, displays the best-fit model.
        Passes *kwargs* to lenstronomy.Sampling.Samplers.pso.‎ParticleSwarmOptimizer.optimize (in particular, 'max_iter' will set the max number of iterations).
        
        Outputs:
        *logL: the maximum log-likelihood value that was found during the optimization
        *kwargs_values_best: the dictionary of all parameter values (linear and non-linear, free and fixed) maximizing the likelihood.
        If return_all is True, also returns *global_best*, the array of free, non-linear parameters maximizing the likelihood (sorted according to the order in the ParamHandler of self.multi_spec), and the PSO chain (list of log-likelihoods and particle positions/velocities).'''

        time_start = time.time()
        global_best, [log_likelihood_list, pos_list, vel_list] = self.pso.optimize(**kwargs)
        logL, kwargs_values_best = self.multi_spec.log_likelihood_from_array(global_best, **self.kwargs_lik)
        time_end = time.time()
        print('Time taken for PSO optimization: ', time_end - time_start)
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values_best, plot=True, print_res_stats=print_res_stats, norm_residuals=norm_residuals,
                                                **self.kwargs_lik)

        if return_all:
            return logL, kwargs_values_best, global_best, [log_likelihood_list, pos_list, vel_list]
        else:
            return logL, kwargs_values_best


class MCMCSampler(Optimizer):
    '''Child class of Optimizer, using a Markov Chain Monte-Carlo algorithm to explore the likelihood.'''
    
    def run_mcmc(self, n_walkers, n_run, n_burn, thin=1, backend_filename=None, start_from_backend=False, skip_initial_state_check=False):
         '''Run MCMC with emcee (see documentation in emcee package for more detail)

         Inputs:
         *n_walkers: number of walkers in the emcee process
         *number of sampling (after burn-in) of the emcee
         * number of burn-in iterations (those will not be saved in the output sample)
         *backend_filename: name of the HDF5 file where sampling state is saved (through emcee backend engine)
         *start_from_backend: bool, if True, start from the state saved in `backup_filename`.
         Otherwise, create a new backup file with name `backup_filename` (any already existing file is overwritten!).
         
         Returns: 
         *dist[ind_best]: the maximum log-likelihood value that was found during the exploration
         *kwargs_values_best: the dictionary of all parameter values (linear and non-linear, free and fixed) maximizing the likelihood.
         *flat_samples: MCMC chain (list of all samples)
         *dist: log-likelihood of samples in the chain
         '''
         self.n_walkers = n_walkers
         num_param_nonlinear = len(self.multi_spec.param_handler.free_nonlinear_param_list)
         norm_lower_limits = (self.bounds_array[:,0] - self.init_array) / self.sigs_array
         norm_upper_limits = (self.bounds_array[:,1] - self.init_array) / self.sigs_array
         
         initpos = np.vstack([self.init_array + self.sigs_array * stats.truncnorm.rvs(norm_lower_limits, norm_upper_limits, size=num_param_nonlinear)
                              for i in range(n_walkers)])         

         if backend_filename is not None:
             backend = emcee.backends.HDFBackend(backend_filename, name="lensqso_specfit_mcmc")
             print("Warning: All samples (including burn-in) will be saved in backup file '{}'.".format(backend_filename))

             if start_from_backend:
                 initpos = None
                 n_run_eff = n_run
             else:
                 n_run_eff = n_burn + n_run
                 backend.reset(n_walkers, num_param_nonlinear)
                 print("Warning: backup file '{}' has been reset!".format(backend_filename))
         else:
             backend = None
             n_run_eff = n_burn + n_run
         
         time_start = time.time()
         print('Computing the MCMC...')
         print('Number of walkers: ', n_walkers)
         print('Burn-in iterations: ', n_burn)
         print('Sampling iterations (in current run):', n_run_eff)
         
         sampler = emcee.EnsembleSampler(n_walkers, num_param_nonlinear, lambda a : self.multi_spec.log_likelihood_from_array(a, **self.kwargs_lik)[0],
                                backend=backend)
         sampler.run_mcmc(initpos, n_run_eff, progress=True, skip_initial_state_check=skip_initial_state_check)
         
         flat_samples = sampler.get_chain(discard=n_burn, thin=thin, flat=True)
         dist = sampler.get_log_prob(flat=True, discard=n_burn, thin=thin)
         time_end = time.time()
         print('Time taken for MCMC sampling: ', time_end - time_start)
         self.samples_mcmc = flat_samples
         self.logL_chain = dist

         #find best fit in chain
         ind_best = np.argmax(dist)
         array_best = flat_samples[ind_best]
         _, kwargs_values_best = self.multi_spec.log_likelihood_from_array(array_best, update=False, **self.kwargs_lik)
         
         return dist[ind_best], kwargs_values_best, flat_samples, dist
    
    def plot_behaviour(self):
         '''Plots the MCMC behaviour to see if the chain has converged.'''
         fig, ax = plt.subplots() 
         num_samples = len(self.samples_mcmc[:, 0])
         n_points = num_samples // self.n_walkers
         param_mcmc = self.multi_spec.param_handler.free_nonlinear_param_list
         for i, param_name in enumerate(param_mcmc):
             samples = self.samples_mcmc[:, i]
             samples_averaged = np.average(samples[: int(n_points * self.n_walkers)].reshape(n_points, self.n_walkers), axis=1)
             end_point = np.mean(samples_averaged)
             samples_renormed = (samples_averaged - end_point) / np.std(samples_averaged)
             ax.plot(samples_renormed, label=param_name)
             dist_averaged = -np.nanmax(self.logL_chain[: int(n_points * self.n_walkers)].reshape(n_points, self.n_walkers),axis=1)
             dist_normed = (dist_averaged - np.nanmax(dist_averaged)) / (
                 np.nanmax(dist_averaged) - np.nanmin(dist_averaged, where = (np.isfinite(dist_averaged)), initial=-np.inf))
         ax.plot(dist_normed, label="logL", color="k", linewidth=2)

    
    def flux_ratio_posterior(self, feature, ref_image='default', samples=None, n_samples=1000, n_burn_add=0):
         '''Calculates a posterior probability distribution on the flux ratios for a given feature, from model parameter samples.
         
         Inputs:
         *feature: label of the emission feature (single line or doublet) used for flux-ratio calculation.
         *ref_image: return the flux ratios relative to this image (by default, the one given as input of the MultiSpectrum() instance).
             If None, return all the possible flux ratios.
         *samples: list of parameter samples. If None, uses the MCMC chain stored in the class (requires a run_mcmc beforehand).
         *n_samples: number of random re-samples of the chain to estimate the posterior
         *n_burn_add: number of samples to ignore at the start of the chain, to avoid re-sampling parts of the chain where the MCMC had not yet converged.
         
         Outputs:
         * feature_fluxratios: array of flux-ratio values
         * list(fr_names.keys()): list of labels for the flux ratios
         '''
         if samples is None:
             samples = self.samples_mcmc
         
         if ref_image=='default':
             ref_image = self.multi_spec.param_handler.ref_image
             
         if not(feature in self.multi_spec.spec_dict[self.multi_spec.param_handler.ref_image].feature_dict):
             raise CustomError('This feature is not in the model.')
        
         num_samples_tot = len(samples[:, 0])
         subsample = n_burn_add + np.random.choice(a=num_samples_tot-n_burn_add, size=n_samples)

         fr_names = {}
         index = 0
         if ref_image is not None:
             assert (ref_image in self.multi_spec.spec_dict.keys())
             for image_name in self.multi_spec.spec_dict: 
                 if not(image_name==ref_image):
                     fr_names['f_'+image_name+'/f_'+ref_image] = index
                     index+=1
         else:
             for image1 in self.multi_spec.spec_dict: 
                 for image2 in self.multi_spec.spec_dict:
                     if not(image1==image2):
                         fr_names['f_'+image1+'/f_'+image2] = index
                         index+=1
              
         #Array that will contain the flux ratios for each sample
         fluxratios_array = np.zeros((n_samples, index))         
                 
         # find position of relevant amplitude parameter in the list of linear params 
         # (depending on whether the Hermite coefficients are treated as linear or non-linear the integrated flux is named differently)
         image_name = self.multi_spec.param_handler.ref_image
         try:
             k = self.multi_spec.spec_dict[image_name].lin_param_handler.linear_param_list.index(feature + '_amp')
         except ValueError:
             k = self.multi_spec.spec_dict[image_name].lin_param_handler.linear_param_list.index(feature + '_coeffsHermite0')
                 
         for i in range(n_samples):
            
             #sample the non-linear parameter
             array_nonlinear_free_params = samples[subsample[i]]

             kwargs_nonlinear_mult = self.multi_spec.param_handler.array2kwargs_nonlinear(array_nonlinear_free_params)
             feature_fluxes = {}
             for image_name in self.multi_spec.spec_dict:
                 mask = self.kwargs_lik['mask_dict'][image_name] if image_name in self.kwargs_lik['mask_dict'] else np.ones_like(
                     self.multi_spec.spec_dict[image_name].lambda_array)
                 lin_params_MLE, lin_params_cov = self.multi_spec.spec_dict[image_name].solve_linear_params(
                     kwargs_nonlinear=kwargs_nonlinear_mult[image_name],mask_array=mask)
                
                 #sample from conditional distribution (linear params knowing non-linear params)
                 lin_params_sample = np.random.multivariate_normal(lin_params_MLE, lin_params_cov, size=1)[0]

                 feature_fluxes[image_name] = lin_params_sample[k]
                
             #calculate the flux-ratios
             if ref_image is not None:
                 for image_name in self.multi_spec.spec_dict: 
                     if not(image_name==ref_image):
                         index = fr_names['f_'+image_name+'/f_'+ref_image]
                         fluxratios_array[i][index] = feature_fluxes[image_name]/feature_fluxes[ref_image]
             else:
                 for image1 in self.multi_spec.spec_dict: 
                     for image2 in self.multi_spec.spec_dict:
                         if not(image1==image2):
                             index = fr_names['f_'+image1+'/f_'+image2]
                             fluxratios_array[i][index] = feature_fluxes[image1]/feature_fluxes[image2]
         
         return fluxratios_array, list(fr_names.keys())   

class FittingSequence():

    '''Class to iteratively run several optimizers. 
    Inputs:
        *fit_param_list: list with entries of the form (type, kwargs) with type in ['COBYQA', 'PSO', 'MCMC', 'update_kwargs_likelihood']
            and kwargs a dictionary corresponding to the format asked by the corresponding class. 
            For 'update_kwargs_likelihood', kwargs needs to be a dictionary with a subset of the expected entries in kwargs_likelihood.
        *multi_spec: instance of MultiSpectrum() class
        *kwargs_likelihood: dictionary of the form {'mask_dict': dictionary of mask arrays for each image, 
                                                    'tol_positive': float}.
        *init_array: array of values for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess during the optimization. If None, the mean values of the priors are used.
        *sigs_array: array of uncertainties for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess for the spread of values during the optimization. If None, the std deviations of the priors are used.
    '''

    def __init__(self, fit_param_list, multi_spec, kwargs_likelihood, init_array=None, sigs_array=None):
        self.fit_param_list = fit_param_list
        self.multi_spec = multi_spec
        self.kwargs_likelihood = kwargs_likelihood

        self.state_params = init_array
        self.state_sigs = sigs_array
        self.chain = []

        assert np.all([step[0] in ['COBYQA', 'PSO', 'MCMC', 'update_kwargs_likelihood'] for step in fit_param_list])

    def run_step(self, type, kwargs, init_array=None, sigs_array=None):
        '''Runs a single step of the fitting sequence.

        Inputs:
        *type: one of the following: 'COBYQA', 'PSO', 'MCMC' or 'update_kwargs_likelihood'
        *kwargs: dictionary with kwargs corresponding to the type of optimizer (see respective classes)
        *init_array: array of values for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess during the optimization. If None, the mean values of the priors are used.
        *sigs_array: array of uncertainties for each free, non-linear parameter (need to be sorted according to the order in the ParamHandler of *multi_spec*).
            Will be used as an initial guess for the spread of values during the optimization. If None, the std deviations of the priors are used.
        '''
        
        if type == 'COBYQA':
            opt = Optimizer(self.multi_spec, self.kwargs_likelihood, init_array, sigs_array)
            logL_best, res_array, kwargs_values_best = opt.find_MLE(plot=False, return_array=True)
            self.state_params = res_array
            self.chain.append(['COBYQA', logL_best, kwargs_values_best])
        elif type == 'PSO':
            pso = PSO(self.multi_spec, kwargs['n_particles'], self.kwargs_likelihood)
            kwargs_pso = copy.deepcopy(kwargs)
            kwargs_pso.pop('n_particles')
            logL_best, kwargs_values_best, global_best, chain = pso.optimize(plot=False, return_all=True, **kwargs_pso)
            self.state_params = global_best
            self.state_sigs = np.std([particle.position for particle in pso.pso.swarm], axis=0) #record spread of final swarm state
            self.chain.append(['PSO', logL_best, kwargs_values_best, chain])
        elif type=='MCMC':
            mcmc_sampler = MCMCSampler(self.multi_spec, self.kwargs_likelihood, init_array, sigs_array)
            logL_best, kwargs_values_best, flat_samples, logL_chain = mcmc_sampler.run_mcmc(**kwargs)
            ind_best = np.argmax(logL_chain)
            self.state_params = flat_samples[ind_best]
            self.state_sigs = np.std(flat_samples, axis=0)
            self.chain.append(['MCMC', logL_best, kwargs_values_best, flat_samples, logL_chain, mcmc_sampler.n_walkers])
        else:
            for key in kwargs:
                self.kwargs_likelihood[key] = kwargs[key]

    def run_sequence(self):
        '''Runs the entire fitting sequence, step by step.
        Returns a list with entries corresponding to the output of each step.
        '''

        try:
            for step in self.fit_param_list:
                self.run_step(type=step[0], kwargs=step[1], init_array=self.state_params, sigs_array=self.state_sigs)
        except ValueError:
            print('All of the initial samples have a likelihood of -np.inf. Try increasing the tol_positive value(s) !')
            return None

        return self.chain

    def plot_convergence(self):
        '''Plots the behaviour during the PSO and MCMC steps in the fitting sequence.'''
        
        for step_run in self.chain:
            if step_run[0] == 'PSO':
                plot_chain(step_run[3], self.multi_spec.param_handler.free_nonlinear_param_list)
            elif step_run[0] == 'MCMC':
                mcmc_sampler = MCMCSampler(self.multi_spec, self.kwargs_likelihood)
                mcmc_sampler.samples_mcmc = step_run[3]
                mcmc_sampler.logL_chain = step_run[4]
                mcmc_sampler.n_walkers = step_run[5]
                mcmc_sampler.plot_behaviour()
            else:
                print('No convergence plot for step: '+ step_run[0])

    def get_kwargs_values_best(self, plot=True, print_res_stats=True, norm_residuals=False):
        '''Returns a dictionary with all the parameter values (free/fixed, linear AND non-linear) that correspond to the best fit that has been found throughout the entire fitting sequence.'''
        step_best = np.argmax([step_run[1] for step_run in self.chain])
        kwargs_values_best = self.chain[step_best][2]
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values_best, plot=True, print_res_stats=print_res_stats, norm_residuals=norm_residuals,
                                                **self.kwargs_likelihood)
        return kwargs_values_best
        



######### add possibility to vary the narrow doublet line ratio ##########

######### recode PSO to avoid lenstronomy dependency ? ##########

######### save chains as .npy files & inputs/results as .json files ?? ##########

######### Priors for linear parameters ? ##########

######### Option to treat some linear parameters as non-linear ? ##########

######### ALLOW PARALLELIZATION ???? ##########

######### LOAD RESULTS FROM A PREVIOUS & POSSIBLY SIMPLER FIT AND SET A PRIOR ##########







