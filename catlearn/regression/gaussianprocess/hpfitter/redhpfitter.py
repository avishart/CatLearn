
import numpy as np
from scipy.optimize import OptimizeResult
from .hpfitter import HyperparameterFitter

class ReducedHyperparameterFitter(HyperparameterFitter):
    def __init__(self,func,optimization_method=None,opt_kwargs={},opt_tr_size=50,**kwargs):
        """ Hyperparameter fitter object with local and global optimization methods for optimizing the hyperparameters on different objective functions.
            The optimizations of the hyperparameters when the training set size is below a number. 
        Parameters:
            func : class
                A class with the objective function used to optimize the hyperparameters.
            optimization_method : class
                A function with the optimization method used.
            opt_kwargs : function 
                A dictionary with the arguments for the optimization method.
            opt_tr_size : int
                A integer that restrict the size of the training set for a hyperparameter optimization.
        """
        super().__init__(func,optimization_method=optimization_method,opt_kwargs=opt_kwargs,**kwargs)
        self.opt_tr_size=opt_tr_size
        
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
        if len(X)<=self.opt_tr_size:
            sol=self.optimization_method(self.func,theta,parameters,model,X,Y,pdis=pdis,**self.opt_kwargs)
        else:
            sol={'fun':np.inf,'x':theta,'hp':hp,'success':False,'nfev':0,'nit':0,'message':"No function values calculated."}
            sol=OptimizeResult(**sol)
        return sol
    
    def copy(self):
        " Copy the hyperparameter fitter. "
        return self.__class__(func=self.func,optimization_method=self.optimization_method,opt_kwargs=self.opt_kwargs,opt_tr_size=self.opt_tr_size)
    
    def __repr__(self):
        return "ReducedHyperparameterFitter(func={},optimization_method={},opt_kwargs={},opt_tr_size={})".format(self.func.__class__.__name__,self.optimization_method.__name__,self.opt_kwargs,self.opt_tr_size)
    
    