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

    def __init__(self, multi_spec, kwargs_likelihood, init_array=None, sigs_array=None):
        '''Docstring TBD. Uses COBYQA method for constrained optimization (needs scipy>=1.14.0).
        multi_spec: instance of MultiSpectrum() class
        kwargs_likelihood: dictionary of the form {'mask_dict': dictionary of mask arrays for each image, 
                                                    'tol_positive': float}.'''
        self.multi_spec = multi_spec
        
        self.multi_spec.set_initial_values()

        # initial values of free (non-linear) parameters + spread of initial samples
        init_array_default, sig_array_default = self.multi_spec.get_init_sample_distrib_free_nonlinear_params() 
        self.init_array = init_array_default if init_array is None else init_array
        self.sigs_array = sig_array_default if sigs_array is None else sigs_array
                                
        self.bounds_array = self.multi_spec.get_bounds_free_nonlinear_params() #prior bounds on free (non-linear) parameters 
        self.kwargs_lik = kwargs_likelihood
        

    def find_MLE(self, plot=True, return_array=False):
        '''Should update the ParamHandler kwargs values automatically'''
        res = minimize(lambda a : -self.multi_spec.log_likelihood_from_array(a, **self.kwargs_lik)[0], 
                       x0 = self.init_array, bounds = self.bounds_array, method='COBYQA')
        
        logL, kwargs_values = self.multi_spec.log_likelihood_from_array(res.x, **self.kwargs_lik)
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values, plot=True, **self.kwargs_lik)

        if return_array:
            return logL, res.x, kwargs_values
        else:
            return logL, kwargs_values



class PSO(Optimizer):
    
    def __init__(self, multi_spec, n_particles, kwargs_likelihood={}):
        super().__init__(multi_spec, kwargs_likelihood)
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

    def optimize(self, plot=True, return_all=False, **kwargs):
        '''Should update the ParamHandler kwargs values automatically'''

        time_start = time.time()
        global_best, [log_likelihood_list, pos_list, vel_list] = self.pso.optimize(**kwargs)
        logL, kwargs_values_best = self.multi_spec.log_likelihood_from_array(global_best, **self.kwargs_lik)
        time_end = time.time()
        print('Time taken for PSO optimization: ', time_end - time_start)
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values_best, plot=True, **self.kwargs_lik)

        if return_all:
            return logL, kwargs_values_best, global_best, [log_likelihood_list, pos_list, vel_list]
        else:
            return logL, kwargs_values_best


class MCMCSampler(Optimizer):
    
     def run_mcmc(self, n_walkers, n_run, n_burn, thin=1, backend_filename=None, start_from_backend=False, skip_initial_state_check=False):
         '''Run MCMC with emcee (see documentation in emcee package for more detail)
         *n_walkers: number of walkers in the emcee process
         *number of sampling (after burn-in) of the emcee
         * number of burn-in iterations (those will not be saved in the output sample)
         *backend_filename: name of the HDF5 file where sampling state is saved (through emcee backend engine)
         *start_from_backend: bool, if True, start from the state saved in `backup_filename`.
         Otherwise, create a new backup file with name `backup_filename` (any already existing file is overwritten!).
         Returns: samples, log-likelihood of samples
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

    
     def narrow_doublet_flux_ratio_posterior(self, narrow_doublet_name, samples=None, n_samples=1000, n_burn_add=0):
         ''' Docstring TBD.'''
         
         if samples is None:
             samples = self.samples_mcmc

         ref_image = self.multi_spec.param_handler.ref_image

         if not(narrow_doublet_name in [doublet[0] for doublet in self.multi_spec.spec_dict[ref_image].kwargs_model['narrow_doublets']]):
             raise CustomError('This narrow doublet is not in the model.')
        
         num_samples_tot = len(samples[:, 0])
         subsample = n_burn_add + np.random.choice(a=num_samples_tot-n_burn_add, size=n_samples)

         #Array that will contain the flux ratios relative to the reference image, for each sample
         fr_names = {}
         index = 0
         for image_name in self.multi_spec.spec_dict: 
             if not(image_name==ref_image):
                 fr_names['f_'+image_name+'/f_'+ref_image] = index
                 index+=1
         narrow_doublet_fluxratios = np.zeros((n_samples, index))
                 
         # find position of relevant amplitude parameter in the list of linear params 
         # (depending on whether the Hermite coefficients are treated as linear or non-linear the integrated flux is named differently)
         try:
             k = self.multi_spec.spec_dict[image_name].lin_param_handler.linear_param_list.index(narrow_doublet_name + '_narrow_doublet_amp')
         except ValueError:
             k = self.multi_spec.spec_dict[image_name].lin_param_handler.linear_param_list.index(narrow_doublet_name + '_narrow_doublet_coeffsHermite0')
                 
         for i in range(n_samples):
             
             #sample the non-linear parameter
             array_nonlinear_free_params = samples[subsample[i]]

             narrow_doublet_fluxes = {} #dictionary with the narrow-line fluxes
             kwargs_nonlinear_mult = self.multi_spec.param_handler.array2kwargs_nonlinear(array_nonlinear_free_params)
             for image_name in self.multi_spec.spec_dict:
                 mask = self.kwargs_lik['mask_dict'][image_name] if image_name in self.kwargs_lik['mask_dict'] else np.ones_like(
                     self.multi_spec.spec_dict[image_name].lambda_array)
                 lin_params_MLE, lin_params_cov = self.multi_spec.spec_dict[image_name].solve_linear_params(kwargs_nonlinear=kwargs_nonlinear_mult[image_name],
                                                                                                            mask_array=mask)
                 #sample from conditional distribution (linear params knowing non-linear params)
                 lin_params_sample = np.random.multivariate_normal(lin_params_MLE, lin_params_cov, size=1)[0]

                 narrow_doublet_fluxes[image_name] = lin_params_sample[k]
                
             #calculate the flux-ratios
             for j, image_name in enumerate(self.multi_spec.spec_dict):
                 if not(image_name==ref_image):
                     index = fr_names['f_'+image_name+'/f_'+ref_image]
                     narrow_doublet_fluxratios[i][index] = narrow_doublet_fluxes[image_name]/narrow_doublet_fluxes[ref_image]

         
         return narrow_doublet_fluxratios, list(fr_names.keys())   
                                                                                                       


class FittingSequence():

    '''Docstring TBD. 
        fit_param_list: list with entries of the form (type, kwargs) with type in ['COBYQA', 'PSO', 'MCMC', 'update_kwargs_likelihood'].
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
        '''Docstring TBD. 
        Runs a single step of the fitting sequence.
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
        '''Docstring TBD. 
        Runs the entire fitting sequence.
        '''
        
        for step in self.fit_param_list:
            self.run_step(type=step[0], kwargs=step[1], init_array=self.state_params, sigs_array=self.state_sigs)

        return self.chain

    def plot_convergence(self):
        '''Docstring TBD.
        '''
        
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

    def get_kwargs_values_best(self, plot=True):
        step_best = np.argmax([step_run[1] for step_run in self.chain])
        kwargs_values_best = self.chain[step_best][2]
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values_best, plot=True, **self.kwargs_likelihood)
        return kwargs_values_best
        




######### recode PSO to avoid lenstronomy dependecy ? ##########

######### save chains as .npy files & inputs/results as .json files ?? ##########

######### Implement Voigt profile ?  ##########

######### Priors for linear parameters ? ##########

######### Option to treat some linear parameters as non-linear ? ##########

######### ALLOW PARALLELIZATION ???? ##########

######### LOAD RESULTS FROM A PREVIOUS & POSSIBLY SIMPLER FIT AND SET A PRIOR ##########







