import numpy as np
from .pdistributions import Prior_distribution
from scipy.special import loggamma

class Invgamma_prior(Prior_distribution):
    def __init__(self,a=1e-20,b=1e-20):
        'Inverse-Gamma distribution'
        self.update(a=a,b=b)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return self.lnpre-2*self.a*x-self.b*np.exp(-2*x)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -2*self.a+2*self.b*np.exp(-2*x)
    
    def update(self,a=None,b=None):
        'Update the parameters of distribution function'
        if a!=None:
            self.a=a
        if b!=None:
            self.b=b
        self.lnpre=self.a*np.log(self.b)-loggamma(self.a)
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        mean,var=np.exp(mean),np.exp(2*np.sqrt(var))
        min_v=mean-np.sqrt(var)*2
        a=min_v
        b=min_v
        self.update(a=a,b=b)
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        b=np.exp(2*min_v)
        a=self.b
        self.update(a=a,b=b)
        return self
    
    def __repr__(self):
        return 'Inv-Gamma({:.4f},{:.4f})'.format(self.a,self.b)
