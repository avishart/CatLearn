import numpy as np

def make_prior(model,parameters,X,Y,prior_dis=None,scale=1):
    " Make prior distribution from educated guesses in log space "
    from ..educated import Educated_guess
    ed_guess=Educated_guess(prior=model.prior,kernel=model.kernel,parameters=parameters)
    parameters_set=sorted(list(set(parameters)))
    if isinstance(scale,(float,int)):
        scale={para:scale for para in parameters_set}
    if prior_dis is None:
        from .normal import Normal_prior
        prior_dis={para:[Normal_prior()] for para in parameters_set}
    else:
        prior_dis={para:prior_dis[para].copy() for para in prior_dis.keys()}
    bounds=ed_guess.bounds(X,Y,parameters_set)
    prior_lp={para:prior_dis[para].min_max(bounds[para][:,0],bounds[para][:,1]) for para in parameters_set if para in prior_dis.keys()}
    return prior_lp

class Prior_distribution:
    def __init__(self,**kwargs):
        """ Prior probability distribution used for each type of hyperparameters. 
            If the type of the hyperparameter is multi dimensional (D) it is given in the axis=-1. 
            If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,D) array. 
        """
        
    def pdf(self,x):
        'Probability density function. '
        return np.exp(self.ln_pdf(x))
    
    def deriv(self,x):
        'The derivative of the probability density function as respect to x'
        return self.pdf(x)*self.ln_deriv(x)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        raise NotImplementedError()
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        raise NotImplementedError()
    
    def update(self,**kwargs):
        'Update the parameters of distribution function'
        raise NotImplementedError()
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        raise NotImplementedError()
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        raise NotImplementedError()
    
    def copy(self):
        " Copy the prior distribution of the hyperparameter. "
        return self.__class__()
    
    def __str__(self):
        return 'Prior()'

    def __repr__(self):
        return 'Prior_distribution()'
