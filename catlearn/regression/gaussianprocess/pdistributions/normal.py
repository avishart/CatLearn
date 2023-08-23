import numpy as np
from .pdistributions import Prior_distribution

class Normal_prior(Prior_distribution):
    def __init__(self,mu=0.0,std=10.0,**kwargs):
        'Independent Normal distribution'
        self.update(mu=mu,std=std)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        if self.nosum:
            return -np.log(self.std)-0.5*np.log(2*np.pi)-0.5*((x-self.mu)/self.std)**2
        return np.sum(-np.log(self.std)-0.5*np.log(2*np.pi)-0.5*((x-self.mu)/self.std)**2,axis=-1)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return -(x-self.mu)/self.std**2
    
    def update(self,mu=None,std=None,**kwargs):
        'Update the parameters of distribution function'
        if mu is not None:
            if isinstance(mu,(float,int)):
                self.mu=mu
            else:
                self.mu=np.array(mu).reshape(-1)
        if std is not None:
            if isinstance(std,(float,int)):
                self.std=std
            else:
                self.std=np.array(std).reshape(-1)
        if isinstance(self.mu,(float,int)) and isinstance(self.std,(float,int)):
            self.nosum=True
        else:
            self.nosum=False
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        return self.update(mu=mean,std=np.sqrt(var))
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        mu=0.5*(min_v+max_v)
        return self.update(mu=mu,std=np.sqrt(2)*(max_v-mu))
    
    def copy(self):
        " Copy the prior distribution of the hyperparameter. "
        return self.__class__(mu=self.mu,std=self.std)
    
    def __str__(self):
        return 'Normal({},{})'.format(self.mu,self.std)
    
    def __repr__(self):
        return 'Normal_prior({},{})'.format(self.mu,self.std**2)
