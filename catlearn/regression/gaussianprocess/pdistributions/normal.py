import numpy as np
from .pdistributions import Prior_distribution

class Normal_prior(Prior_distribution):
    def __init__(self,mu=0.0,std=10.0,**kwargs):
        """ 
        Independent Normal prior distribution used for each type of hyperparameters in log-space. 
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        Parameters:
            mu: float or (H) array
                The mean of the normal distribution. 
            std: float or (H) array
                The standard deviation of the normal distribution.
        """
        self.update_arguments(mu=mu,
                              std=std,
                              **kwargs)
    
    def ln_pdf(self,x):
        if self.nosum:
            return -np.log(self.std)-0.5*np.log(2.0*np.pi)-0.5*((x-self.mu)/self.std)**2
        return np.sum(-np.log(self.std)-0.5*np.log(2.0*np.pi)-0.5*((x-self.mu)/self.std)**2,axis=-1)
    
    def ln_deriv(self,x):
        return -(x-self.mu)/self.std**2
    
    def update_arguments(self,mu=None,std=None,**kwargs):
        """
        Update the object with its arguments. The existing arguments are used if they are not given.
        Parameters:
            mu: float or (H) array
                The mean of the normal distribution. 
            std: float or (H) array
                The standard deviation of the normal distribution.
        Returns:
            self: The updated object itself.
        """
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
        return self.update_arguments(mu=mean,std=np.sqrt(var))
    
    def min_max(self,min_v,max_v):
        mu=0.5*(min_v+max_v)
        return self.update_arguments(mu=mu,std=np.sqrt(2.0)*(max_v-mu))
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mu=self.mu,std=self.std)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
