
import numpy as np

class HyperparameterFitter:
    def __init__(self,func,optimization_method=None,opt_kwargs={},**kwargs):
        """ Hyperparameter fitter object with local and global optimization methods for optimizing the hyperparameters on different objective functions. 
        Parameters:
            func : class
                A class with the objective function used to optimize the hyperparameters.
            optimization_method : function
                A function with the optimization method used.
            opt_kwargs : dict 
                A dictionary with the arguments for the optimization method.
        """
        self.func=func.copy()
        if optimization_method is None:
            from ..optimizers.global_opt import local_optimize
            optimization_method=local_optimize
        self.optimization_method=optimization_method
        self.opt_kwargs=opt_kwargs.copy()
        
    def fit(self,X,Y,model,hp=None,pdis=None,**kwargs):
        """ Optimize the hyperparameters 
        Parameters:
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            model : Model
                The Machine Learning Model with kernel and prior that are optimized.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
        """
        if hp is None:
            hp=model.get_hyperparams()
        theta,parameters=self.hp_to_theta(hp)
        model=model.copy()
        self.func.reset_solution()
        sol=self.optimization_method(self.func,theta,parameters,model,X,Y,pdis=pdis,**self.opt_kwargs)
        return sol
    
    def hp_to_theta(self,hp):
        " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
        parameters_set=sorted(hp.keys())
        theta=sum([list(hp[para]) for para in parameters_set],[])
        parameters=sum([[para]*len(hp[para]) for para in parameters_set],[])
        return np.array(theta),parameters
    
    def copy(self):
        " Copy the hyperparameter fitter. "
        return self.__class__(func=self.func,optimization_method=self.optimization_method,opt_kwargs=self.opt_kwargs)
    
    def __repr__(self):
        return "HyperparameterFitter(func={},optimization_method={},opt_kwargs={})".format(self.func.__class__.__name__,self.optimization_method.__name__,self.opt_kwargs)
    