
import numpy as np
import copy

class HyperparameterFitter:
    def __init__(self,func,optimization_method=None,opt_kwargs={},distance_matrix=True):
        """ Object with local and global optimization methods for optimizing the hyperparameters on different object functions 
        Parameters:
            func : class
                A class with the object function used to optimize the hyperparameters.
            optimization_method : class
                A function with the optimization method used.
            opt_kwargs : dict 
                A dictionary with the arguments for the optimization method.
            distance_matrix : bool
                Whether to reuse the distance matrix for the optimization.
        """
        self.func=copy.deepcopy(func)
        self.optimization_method=optimization_method
        self.opt_kwargs=opt_kwargs
        self.distance_matrix=distance_matrix
        
    def fit(self,X,Y,GP,hp=None,prior=None):
        """ Optimize the hyperparameters 
        Parameters:
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            GP : GaussianProcess
                The Gaussian Process with kernel and prior that are optimized.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            prior : dict
                A dict of prior distributions for each hyperparameter.
        """
        if hp is None:
            hp=GP.hp.copy()
        theta,parameters=self.hp_to_theta(hp)
        gp=copy.deepcopy(GP)
        # Whether to use distance matrix
        dis_m=gp.kernel.distances(X) if self.distance_matrix else None
        sol=self.optimization_method(self.func,theta,gp,parameters,X,Y,prior=prior,dis_m=dis_m,**self.opt_kwargs)
        return sol
    
    def hp_to_theta(self,hp):
        " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
        parameters_set=sorted(set(hp.keys()))
        theta=[list(np.array(hp[para]).reshape(-1)) for para in parameters_set]
        parameters=sum([[para]*len(theta[p]) for p,para in enumerate(parameters_set)],[])
        theta=np.array(sum(theta,[]))
        return theta,parameters 
    