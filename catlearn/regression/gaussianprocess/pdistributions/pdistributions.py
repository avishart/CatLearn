import numpy as np

class Prior_distribution:
    def __init__(self,**kwargs):
        """ 
        Prior probability distribution used for each type of hyperparameters in log-space. 
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        """
        self.update_arguments(**kwargs)
        
    def pdf(self,x):
        """ 
        Probability density function.
        Parameter:
            x: float or (M,H) array
                x is the hyperparameter value used by the prior distribution.
                x can be a float if the prior distribution only consider 
                a single hyperparameter of that type.
                x can be a (1,H) array if the prior distribution consider 
                H hyperparameter of that type.  
                x can be a (M,H) array if the prior distribution consider 
                H hyperparameter of that type with M different values.  
        Returns: 
            float: Value of the probability density function.
            or
            (M) array: M values of the probability density function if M different values is given.
        """
        return np.exp(self.ln_pdf(x))
    
    def deriv(self,x):
        " The derivative of the probability density function as respect to x. "
        return self.pdf(x)*self.ln_deriv(x)
    
    def ln_pdf(self,x):
        """ 
        Log of probability density function.
        Parameter:
            x: float or (M,H) array
                x is the hyperparameter value used by the prior distribution.
                x can be a float if the prior distribution only consider 
                a single hyperparameter of that type.
                x can be a (1,H) array if the prior distribution consider 
                H hyperparameter of that type.  
                x can be a (M,H) array if the prior distribution consider 
                H hyperparameter of that type with M different values.  
        Returns: 
            float: Value of the log of probability density function.
            or
            (M) array: M values of the log of probability density function if M different values is given.
        """
        raise NotImplementedError()
    
    def ln_deriv(self,x):
        " The derivative of the log of the probability density function as respect to x. "
        raise NotImplementedError()
    
    def update_arguments(self,**kwargs):
        """
        Update the object with its arguments. The existing arguments are used if they are not given.
        Returns:
            self: The updated object itself.
        """
        return self
            
    def mean_var(self,mean,var):
        " Obtain the parameters of the distribution function by the mean and variance values. "
        raise NotImplementedError()
    
    def min_max(self,min_v,max_v):
        " Obtain the parameters of the distribution function by the minimum and maximum values. "
        raise NotImplementedError()
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict()
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs

    def copy(self):
        " Copy the object. "
        # Get all arguments
        arg_kwargs,constant_kwargs,object_kwargs=self.get_arguments()
        # Make a clone
        clone=self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key,value in constant_kwargs.items():
                clone.__dict__[key]=value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key,value in object_kwargs.items():
                clone.__dict__[key]=value.copy()
        return clone
    
    def __repr__(self):
        arg_kwargs=self.get_arguments()[0]
        str_kwargs=",".join([f"{key}={value}" for key,value in arg_kwargs.items()])
        return "{}({})".format(self.__class__.__name__,str_kwargs)
