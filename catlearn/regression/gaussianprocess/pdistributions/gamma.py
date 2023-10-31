import numpy as np
from .pdistributions import Prior_distribution
from scipy.special import loggamma

class Gamma_prior(Prior_distribution):
    def __init__(self,a=1e-20,b=1e-20,**kwargs):
        """ 
        Gamma prior distribution used for each type of hyperparameters in log-space. 
        The Gamma distribution is variable transformed from linear- to log-space.
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        Parameters:
            a: float or (H) array
                The shape parameter. 
            b: float or (H) array
                The scale parameter.
        """
        self.update_arguments(a=a,
                              b=b,
                              **kwargs)
    
    def ln_pdf(self,x):
        if self.nosum:
            return self.lnpre+2.0*self.a*x-self.b*np.exp(2.0*x)
        return np.sum(self.lnpre+2.0*self.a*x-self.b*np.exp(2.0*x),axis=-1)

    def ln_deriv(self,x):
        return 2.0*self.a-2.0*self.b*np.exp(2.0*x)
    
    def update_arguments(self,a=None,b=None,**kwargs):
        """
        Update the object with its arguments. The existing arguments are used if they are not given.
        Parameters:
            a: float or (H) array
                The shape parameter. 
            b: float or (H) array
                The scale parameter.
        Returns:
            self: The updated object itself.
        """
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
        self.lnpre=np.log(2.0)+self.a*np.log(self.b)-loggamma(self.a)
        if isinstance(self.a,(float,int)) and isinstance(self.b,(float,int)):
            self.nosum=True
        else:
            self.nosum=False
        return self
            
    def mean_var(self,mean,var):
        mean,var=np.exp(mean),np.exp(2.0*np.sqrt(var))
        a=mean**2.0/var
        if a==0:
            a=1
        return self.update_arguments(a=a,b=mean/var)

    def min_max(self,min_v,max_v):
        min_v,max_v=np.exp(min_v),np.exp(max_v)
        mean=0.5*(min_v+max_v)
        var=0.5*(max_v-min_v)**2
        return self.update_arguments(a=mean**2/var,b=mean/var)

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(a=self.a,b=self.b)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
