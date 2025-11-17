import numpy as np
import matplotlib.pyplot as plt
from multi_spectrum import MultiSpectrum
import copy
from scipy.optimize import minimize
from math import floor
import math
from tqdm import tqdm
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer, Particle

class Optimizer():

    def __init__(self, multi_spec, kwargs_likelihood={}):
        '''Docstring TBD. Uses COBYQA method for constrained optimization (needs scipy>=1.14.0).
        multi_spec: instance of MultiSpectrum() class'''
        self.multi_spec = multi_spec
        
        self.multi_spec.set_initial_values()
        self.init_array, self.sigs_array = self.multi_spec.get_init_sample_distrib_free_nonlinear_params() 
                                # initial values of free (non-linear) parameters + spread of initial samples
        self.bounds_array = self.multi_spec.get_bounds_free_nonlinear_params() #prior bounds on free (non-linear) parameters 
        self.kwargs_lik = kwargs_likelihood
        

    def find_MLE(self, plot=True):
        '''Should update the ParamHandler kwargs values automatically'''
        res = minimize(lambda a : -self.multi_spec.log_likelihood_from_array(a, **self.kwargs_lik)[0], 
                       x0 = self.init_array, bounds = self.bounds_array, method='COBYQA')
        
        logL, kwargs_values = self.multi_spec.log_likelihood_from_array(res.x, **self.kwargs_lik)
        if plot:
            _ = self.multi_spec.simulateSpectra(kwargs_values, plot=True, **self.kwargs_lik)

        return logL, kwargs_values



class PSO(Optimizer):
    
    def __init__(self, multi_spec, n_particles):
        super().__init__(multi_spec)
        self.pso = ParticleSwarmOptimizer(func=self.multi_spec.log_likelihood_from_array, particle_count = n_particles,
                                          low=self.bounds_array[:,0], high=self.bounds_array[:,1])

        ## Initialize with samples (override uniform intialization from lenstronomy)
        swarm = []
        for _ in range(self.pso.particleCount):
            swarm.append(Particle(np.random.normal(self.init_array, self.sigs_array, size=self.pso.param_count), np.zeros(self.pso.param_count)))
        self.pso.swarm = swarm

    def optimize(self, **kwargs):
        '''Should update the ParamHandler kwargs values automatically'''
        return self.pso.optimize(**kwargs)


######### TBD: UPDATE PSO  ##########

######### Implement Voigt profile ?  ##########

######### TBD: MCMC SAMPLER ##########

######### ALLOW PARALLELIZATION ???? ##########

######### LOAD RESULTS FROM A PREVIOUS & POSSIBLY SIMPLER FIT AND SET A PRIOR ##########







