import numpy as np
from .pdistributions import Prior_distribution
from scipy.special import loggamma

class Gamma_prior(Prior_distribution):
    def __init__(self,a=1e-20,b=1e-20):
        'Gamma distribution'
        self.update(a=a,b=b)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return self.lnpre+self.a*x-self.b*np.exp(x)

    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return self.a-self.b*np.exp(x)
    
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
        a=mean**2/var
        if a==0:
            a=1
        b=mean/var
        self.update(a=a,b=b)
        return self

    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        min_v,max_v=np.exp(min_v),np.exp(max_v)
        mean=0.5*(min_v+max_v)
        var=0.5*(max_v-min_v)**2
        a=mean**2/var
        b=mean/var
        self.update(a=a,b=b)
        return self
    
    def __repr__(self):
        return 'Gamma({:.4f},{:.4f})'.format(self.a,self.b)