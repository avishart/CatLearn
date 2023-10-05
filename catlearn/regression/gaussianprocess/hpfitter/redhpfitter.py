
import numpy as np
from scipy.optimize import OptimizeResult
from .hpfitter import HyperparameterFitter

class ReducedHyperparameterFitter(HyperparameterFitter):
    def __init__(self,func,optimizer=None,bounds=None,use_update_pdis=False,get_prior_mean=False,use_stored_sols=False,opt_tr_size=50,**kwargs):
        """ 
        Hyperparameter fitter object with an optimizer for optimizing the hyperparameters on different given objective functions. 
        The optimization of the hyperparameters are only performed when the training set size is below a number. 
        Parameters:
            func : ObjectiveFunction class
                A class with the objective function used to optimize the hyperparameters.
            optimizer : Optimizer class
                A class with the used optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                Most of the global optimizers are using boundary conditions. 
                The bounds in this class will be used for the optimizer and func.
            use_update_pdis : bool
                Whether to update the prior distributions of the hyperparameters with the given boundary conditions.
            get_prior_mean : bool
                Whether to get the parameters of the prior mean in the solution.
            use_stored_sols : bool
                Whether to store the solutions.
            opt_tr_size: int
                The maximum size of the training set before the hyperparameters are not optimized.
        """
        super().__init__(func,
                         optimizer=optimizer,
                         bounds=bounds,
                         use_update_pdis=use_update_pdis,
                         get_prior_mean=get_prior_mean,
                         use_stored_sols=use_stored_sols,
                         opt_tr_size=opt_tr_size,
                         **kwargs)
        
    def fit(self,X,Y,model,hp=None,pdis=None,**kwargs):
        """ 
        Optimize the hyperparameters 
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
        Returns: 
            dict : A solution dictionary with objective function value, optimized hyperparameters,
                success statement, and number of used evaluations.
        """
        # Check if optimization is needed
        if len(X)<=self.opt_tr_size:
            # Optimize the hyperparameters
            return super().fit(X,Y,model,hp=hp,pdis=pdis,**kwargs)
        # Use existing hyperparameters
        hp,theta,parameters=self.get_hyperparams(hp,model)
        # Do not optimize hyperparameters
        sol={'fun':np.inf,'x':theta,'hp':hp,
             'success':False,'nfev':0,'nit':0,
             'message':"No function values calculated."}
        sol=OptimizeResult(**sol)
        # Get the full set of hyperparameters in the model
        sol=self.get_full_hp(sol,model)
        return sol
    
    def update_arguments(self,func=None,optimizer=None,bounds=None,use_update_pdis=None,get_prior_mean=None,use_stored_sols=None,opt_tr_size=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            func : ObjectiveFunction class
                A class with the objective function used to optimize the hyperparameters.
            optimizer : Optimizer class
                A class with the used optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                Most of the global optimizers are using boundary conditions. 
                The bounds in this class will be used for the optimizer and func.
            use_update_pdis : bool
                Whether to update the prior distributions of the hyperparameters with the given boundary conditions.
            get_prior_mean : bool
                Whether to get the parameters of the prior mean in the solution.
            use_stored_sols : bool
                Whether to store the solutions.
            opt_tr_size: int
                The maximum size of the training set before the hyperparameters are not optimized.
        Returns:
            self: The updated object itself.
        """
        if func is not None:
            self.func=func.copy()
        if optimizer is not None:
            self.optimizer=optimizer.copy()
        if bounds is not None:
            self.bounds=bounds.copy()
        if use_update_pdis is not None:
            self.use_update_pdis=use_update_pdis
        if get_prior_mean is not None:
            self.get_prior_mean=get_prior_mean
        if use_stored_sols is not None:
            self.use_stored_sols=use_stored_sols
        if opt_tr_size is not None:
            self.opt_tr_size=opt_tr_size
        # Empty the stored solutions
        self.sols=[]
        # Make sure that the objective function gets the prior mean parameters or not
        self.func.update_arguments(get_prior_mean=self.get_prior_mean)
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(func=self.func,
                        optimizer=self.optimizer,
                        bounds=self.bounds,
                        use_update_pdis=self.use_update_pdis,
                        get_prior_mean=self.get_prior_mean,
                        use_stored_sols=self.use_stored_sols,
                        opt_tr_size=self.opt_tr_size)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict(sols=self.get_sols())
        return arg_kwargs,constant_kwargs,object_kwargs
    