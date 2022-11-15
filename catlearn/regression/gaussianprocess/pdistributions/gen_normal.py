import numpy as np
from .pdistributions import Prior_distribution

class Gen_normal_prior(Prior_distribution):
    def __init__(self,mu=0,s=10,v=2):
        'Generalized normal distribution'
        self.mu=mu
        self.s=s
        self.v=v
    
    def ln_pdf(self,x):
        'Log of probability density function'
        return -((x-self.mu)/self.s)**(2*self.v)-np.log(self.s)+np.log(0.52)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return (-(2*self.v)*((x-self.mu)**(2*self.v-1)))/(self.s**(2*self.v))
    
    def update(self,mu=None,s=None,v=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if s!=None:
            self.s=s
        if v!=None:
            self.v=v
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        self.mu=mean
        self.s=np.sqrt(var/0.32)
        return self
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        self.mu=(max_v+min_v)/2
        self.s=np.sqrt(2/0.32)*(max_v-self.mu)
        return self
    
    def __repr__(self):
        return 'Generalized-normal({:.4f},{:.4f},{})'.format(self.mu,self.s,self.v)
