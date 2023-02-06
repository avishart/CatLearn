import numpy as np
import copy

def make_prior(GP,parameters,X,Y,prior_dis=None,scale=1):
    " Make prior distribution from educated guesses in log space "
    from ..educated import Educated_guess
    ed_guess=Educated_guess(GP)
    prior_lp={}
    parameters_set=sorted(list(set(parameters)))
    if isinstance(scale,(float,int)):
        scale={para:scale for para in parameters_set}
    if prior_dis is None:
        from .normal import Normal_prior
        prior_dis={para:[Normal_prior()] for para in parameters_set}
    bounds=ed_guess.bounds(X,Y,parameters_set)
    for para in parameters_set:
        if para in prior_dis.keys():
            prior_d=copy.deepcopy(prior_dis[para])
            if len(prior_d)!=len(bounds[para]):
                prior_d=[prior_d[0]]*len(bounds[para])
            prior_lp[para]=np.array([prior_d[b].min_max(bound[0],bound[1]) for b,bound in enumerate(bounds[para])])
    return prior_lp

class Prior_distribution:
    def __init__(self):
        """ Prior probability distribution used for the hyperparameters """
        
    def pdf(self,x):
        'Probability density function'
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
    
    def update(self,start=None,end=None,prob=None):
        'Update the parameters of distribution function'
        raise NotImplementedError()
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        raise NotImplementedError()
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        raise NotImplementedError()

    def copy(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        return 'Prior()'
