import numpy as np
from .pdistributions import Prior_distribution

class Normal_prior(Prior_distribution):
    def __init__(self,mu=0,std=10):
        'Normal distribution'
        self.mu=mu
        self.std=std
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return -np.log(self.std)-0.5*np.log(2*np.pi)-0.5*((x-self.mu)/self.std)**2
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -(x-self.mu)/self.std**2
    
    def update(self,mu=None,std=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if std!=None:
            self.std=std
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=mean
        self.std=np.sqrt(var)
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.mu=np.nanmean([min_v,max_v])
        self.std=np.sqrt(2)*(max_v-self.mu)
        return self
    
    def __repr__(self):
        return 'Normal({:.4f},{:.4f})'.format(self.mu,self.std**2)
