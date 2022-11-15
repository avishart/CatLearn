import numpy as np
from .pdistributions import Prior_distribution

class Uniform_prior(Prior_distribution):
    def __init__(self,start=-18,end=18,prob=1):
        'Uniform distribution'
        self.start=start
        self.end=end
        self.prob=prob
    
    def ln_pdf(self,x):
        'Log of probability density function'
        ln_0=-np.log(np.nan_to_num(np.inf))
        return np.where(x>=self.start,np.where(x<=self.end,np.log(self.prob),ln_0),ln_0)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return 0*x
    
    def update(self,start=None,end=None,prob=None):
        'Update the parameters of distribution function'
        if start!=None:
            self.start=start
        if end!=None:
            self.end=end
        if prob!=None:
            self.prob=prob
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        std=np.sqrt(var)
        self.start=mean-4*std
        self.end=mean+4*std
        self.prob=1/(self.end-self.start)
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.start=min_v
        self.end=max_v
        self.prob=1/(self.end-self.start)
        return self
    
    def __repr__(self):
        return 'Uniform({:.4f},{:.4f},{:.4f)'.format(self.start,self.end,self.prob)
