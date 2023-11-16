import numpy as np

class HyperparameterFitter:
    def __init__(self,func,optimizer=None,bounds=None,use_update_pdis=False,get_prior_mean=False,use_stored_sols=False,**kwargs):
        """ 
        Hyperparameter fitter object with an optimizer for optimizing the hyperparameters on different given objective functions. 
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
        """
        # Set the default optimizer
        if optimizer is None:
            from ..optimizers.optimizer import FunctionEvaluation
            optimizer=FunctionEvaluation(jac=False)
        # Set the default boundary conditions
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            self.bounds=VariableTransformation(bounds=None)
        # Set all the arguments
        self.update_arguments(func=func,
                              optimizer=optimizer,
                              bounds=bounds,
                              use_update_pdis=use_update_pdis,
                              get_prior_mean=get_prior_mean,
                              use_stored_sols=use_stored_sols,
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
        # Copy the model so it is not changed outside of the optimization
        model=self.copy_model(model)
        # Always reset the solution in the objective function
        self.reset_func()
        # Get hyperparameters
        hp,theta,parameters=self.get_hyperparams(hp,model)
        # Update bounds
        self.update_bounds(model,X,Y,parameters)
        # Update prior distributions of hyperparameters
        pdis=self.update_pdis(pdis,model,X,Y,parameters)
        # Modify the hyperparameters
        theta,parameters=self.modify_hyperparams(hp)
        # Optimize the hyperparameters
        sol=self.optimizer.run(self.func,theta,parameters,model,X,Y,pdis=pdis)
        # Get the full set of hyperparameters in the model
        sol=self.get_full_hp(sol,model)
        # Store the solution 
        self.store_sol(sol)
        return sol
    
    def update_arguments(self,func=None,optimizer=None,bounds=None,use_update_pdis=None,get_prior_mean=None,use_stored_sols=None,**kwargs):
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
        # Empty the stored solutions
        self.sols=[]
        # Make sure that the objective function gets the prior mean parameters or not
        self.func.update_arguments(get_prior_mean=self.get_prior_mean)
        return self

    def copy_model(self,model,**kwargs):
        " Make a copy of the model, so it is not overwritten. "
        return model.copy()
    
    def reset_func(self,**kwargs):
        " Reset the solution in objective function object. "
        return self.func.reset_solution()
    
    def get_hyperparams(self,hp,model,**kwargs):
        " Get default hyperparameters if they are not given. "
        if hp is None:
            # Get the hyperparameters from the model
            hp=model.get_hyperparams()
        # Get the values and hyperparameter names
        theta,parameters=self.hp_to_theta(hp)
        return hp,theta,parameters
    
    def modify_hyperparams(self,hp,**kwargs):
        " Modify the hyperparameters by checking what parameters is used by the objective function. "
        if self.func.use_analytic_prefactor:
            # Remove the prefactor hyperparameter if it is solved internally 
            hp.pop('prefactor',None)
        if self.func.use_optimized_noise:
            # Remove the relative-noise hyperparameter if it is solved internally 
            hp.pop('noise',None)
        theta,parameters=self.hp_to_theta(hp)
        return theta,parameters
    
    def hp_to_theta(self,hp):
        " Transform a dictionary of hyperparameters to a list of values and a list of hyperparameter names. " 
        parameters_set=sorted(hp.keys())
        theta=sum([list(hp[para]) for para in parameters_set],[])
        parameters=sum([[para]*len(hp[para]) for para in parameters_set],[])
        return np.array(theta),parameters
    
    def update_bounds(self,model,X,Y,parameters,**kwargs):
        " Update the boundary condition class with the data. "
        # Give the stored solutions to the bounds if used
        if self.use_stored_sols:
            self.bounds.update_arguments(sols=self.get_sols())
        # Update the boundary conditions
        self.bounds.update_bounds(model,X,Y,parameters)
        # Update the bounds in the objective function
        self.func.update_arguments(bounds=self.bounds)
        # Update the bounds in the optimizer object
        self.optimizer.update_arguments(bounds=self.bounds)
        return self.bounds
    
    def update_pdis(self,pdis,model,X,Y,parameters,**kwargs):
        " Update the prior distributions of the hyperparameters with the boundary conditions. "
        if self.use_update_pdis and pdis is not None:
            from ..pdistributions.update_pdis import update_pdis
            pdis=update_pdis(model,parameters,X,Y,bounds=self.bounds,pdis=pdis)
        return pdis
    
    def get_full_hp(self,sol,model,**kwargs):
        " Get the full hyperparameter dictionary with hyperparameters that are optimized and within the model. "
        sol['full hp']=model.get_hyperparams()
        sol['full hp'].update(sol['hp'])
        return sol

    def store_sol(self,sol,**kwargs):
        " Store the solutions. "
        if self.use_stored_sols:
            self.sols.append(sol)
        return self.sols

    def get_sols(self,**kwargs):
        " Get the stored solutions. "
        return self.sols.copy()
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(func=self.func,
                        optimizer=self.optimizer,
                        bounds=self.bounds,
                        use_update_pdis=self.use_update_pdis,
                        get_prior_mean=self.get_prior_mean,
                        use_stored_sols=self.use_stored_sols)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict(sols=self.get_sols())
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
    