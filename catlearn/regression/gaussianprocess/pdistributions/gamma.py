import numpy as np
from .pdistributions import Prior_distribution
from scipy.special import loggamma

class Gamma_prior(Prior_distribution):
    def __init__(self,a=1e-20,b=1e-20,**kwargs):
        'Gamma distribution'
        self.update(a=a,b=b)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        if self.nosum:
            return self.lnpre+self.a*x-self.b*np.exp(x)
        return np.sum(self.lnpre+self.a*x-self.b*np.exp(x),axis=-1)

    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return self.a-self.b*np.exp(x)
    
    def update(self,a=None,b=None,**kwargs):
        'Update the parameters of distribution function'
        if a is not None:
            if isinstance(a,(float,int)):
                self.a=a
            else:
                self.a=np.array(a).reshape(-1)
        if b is not None:
            if isinstance(b,(float,int)):
                self.b=b
            else:
                self.b=np.array(b).reshape(-1)
        self.lnpre=self.a*np.log(self.b)-loggamma(self.a)
        if isinstance(self.a,(float,int)) and isinstance(self.b,(float,int)):
            self.nosum=True
        else:
            self.nosum=False
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        mean,var=np.exp(mean),np.exp(2*np.sqrt(var))
        a=mean**2/var
        if a==0:
            a=1
        return self.update(a=a,b=mean/var)

    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        min_v,max_v=np.exp(min_v),np.exp(max_v)
        mean=0.5*(min_v+max_v)
        var=0.5*(max_v-min_v)**2
        return self.update(a=mean**2/var,b=mean/var)
    
    def copy(self):
        " Copy the prior distribution of the hyperparameter. "
        return self.__class__(a=self.a,b=self.b)
    
    def __str__(self):
        return 'Gamma_prior({},{})'.format(self.a,self.b)

    def __repr__(self):
        return 'Gamma({},{})'.format(self.a,self.b)
