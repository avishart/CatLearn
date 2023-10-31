import numpy as np
from .pdistributions import Prior_distribution

class Uniform_prior(Prior_distribution):
    def __init__(self,start=-18.0,end=18.0,prob=1.0,**kwargs):
        """ 
        Uniform prior distribution used for each type of hyperparameters in log-space. 
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        Parameters:
            start: float or (H) array
                The start of non-zero prior distribution value of the hyperparameter in log-space. 
            end: float or (H) array
                The end of non-zero prior distribution value of the hyperparameter in log-space. 
            prob: float or (H) array
                The non-zero prior distribution value.
        """
        self.update_arguments(start=start,
                              end=end,
                              prob=prob,
                              **kwargs)
    
    def ln_pdf(self,x):
        ln_0=-np.log(np.nan_to_num(np.inf))
        if self.nosum:
            return np.where(x>=self.start,np.where(x<=self.end,np.log(self.prob),ln_0),ln_0)
        return np.sum(np.where(x>=self.start,np.where(x<=self.end,np.log(self.prob),ln_0),ln_0),axis=-1)
    
    def ln_deriv(self,x):
        return 0.0*x
    
    def update_arguments(self,start=None,end=None,prob=None,**kwargs):
        """
        Update the object with its arguments. The existing arguments are used if they are not given.
        Parameters:
            start: float or (H) array
                The start of non-zero prior distribution value of the hyperparameter in log-space. 
            end: float or (H) array
                The end of non-zero prior distribution value of the hyperparameter in log-space. 
            prob: float or (H) array
                The non-zero prior distribution value.
        Returns:
            self: The updated object itself.
        """
        if start is not None:
            if isinstance(start,(float,int)):
                self.start=start
            else:
                self.start=np.array(start).reshape(-1)
        if end is not None:
            if isinstance(end,(float,int)):
                self.end=end
            else:
                self.end=np.array(end).reshape(-1)
        if prob is not None:
            if isinstance(prob,(float,int)):
                self.prob=prob
            else:
                self.prob=np.array(prob).reshape(-1)
        if isinstance(self.start,(float,int)) and isinstance(self.end,(float,int)) and isinstance(self.prob,(float,int)):
            self.nosum=True
        else:
            self.nosum=False
        return self
            
    def mean_var(self,mean,var):
        std=np.sqrt(var)
        return self.update_arguments(start=mean-4.0*std,end=mean+4.0*std,prob=1.0/(8.0*std))
    
    def min_max(self,min_v,max_v):
        return self.update_arguments(start=min_v,end=max_v,prob=1.0/(max_v-min_v))
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(start=self.start,end=self.start,prob=self.prob)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
