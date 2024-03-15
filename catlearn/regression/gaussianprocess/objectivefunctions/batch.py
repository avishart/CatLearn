import numpy as np
from .objectivefunction import ObjectiveFuction
from ..means.constant import Prior_constant

class BatchFuction(ObjectiveFuction):
    def __init__(self,func,get_prior_mean=False,batch_size=25,equal_size=False,use_same_prior_mean=True,seed=1,**kwargs):
        """ 
        The objective function that is used to optimize the hyperparameters. 
        The instance splits the training data into batches.
        A given objective function is then used as an objective function for the batches.
        The function values from each batch are summed.
        BatchFuction is not recommended for analytic prefactor or noise optimized objective functions!

        Parameters:
            func : ObjectiveFunction class
                A class with the objective function used to optimize the hyperparameters.
            get_prior_mean : bool
                Whether to get the parameters of the prior mean in the solution.
            equal_size : bool
                Whether the clusters are forced to have the same size.
            use_same_prior_mean : bool
                Whether to use the same prior mean for all models.
            seed : int (optional)
                The random seed used to permute the indicies. 
                If seed=None or False or 0, a random seed is not used.
        """
        # Set a random seed
        self.seed=seed
        # Set the arguments
        self.update_arguments(func=func,
                              get_prior_mean=get_prior_mean,
                              batch_size=batch_size,
                              equal_size=equal_size,
                              use_same_prior_mean=use_same_prior_mean,
                              seed=seed,
                              **kwargs)

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        # Get number of data points
        n_data=len(X)
        # Return a single function evaluation if the data set is a batch size
        if n_data<=self.batch_size:
            output=self.func.function(theta,parameters,model,X,Y,pdis=pdis,jac=jac,**kwargs)
            self.sol=self.func.sol
            return output
        # Update the model with hyperparameters and prior mean
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update_model(model,hp)
        self.set_same_prior_mean(model,X,Y)
        # Calculate the number of batches
        n_batches=self.get_number_batches(n_data)
        indicies=np.arange(n_data)
        i_batches=self.randomized_batches(indicies,n_data,n_batches,**kwargs)
        n_full_data=len(np.array(i_batches).reshape(-1))
        # Sum function values together from batches
        fvalue=0.0
        deriv=0.0
        for i_batch in i_batches:
            # Get the feature and target batch
            X_split=X[i_batch]
            Y_split=Y[i_batch]
            # Reset solution so results can be extracted
            self.func.reset_solution()
            # Evaluate the function
            f=self.func.function(theta,parameters,model,X_split,Y_split,pdis=None,jac=jac,**kwargs)
            if jac:
                f,d=f
                deriv+=d
            fvalue+=f
            # Extract the hp from the solution
            hp=self.extract_hp_sol(hp,n_full_data,len(i_batch),**kwargs)
        # Evaluate with prior distribution
        fvalue=fvalue-self.logpriors(hp,pdis,jac=False)
        if jac:
            deriv=deriv-self.logpriors(hp,pdis,jac=True)
            self.update_solution(fvalue,theta,hp,model,jac=jac,deriv=deriv)
            return fvalue,deriv
        self.update_solution(fvalue,theta,hp,model,jac=False)
        return fvalue
    
    def update_arguments(self,func=None,get_prior_mean=None,batch_size=None,equal_size=None,use_same_prior_mean=None,seed=None,**kwargs):
        """
        Update the objective function with its arguments. The existing arguments are used if they are not given.

        Parameters:
            func : ObjectiveFunction class
                A class with the objective function used to optimize the hyperparameters.
            get_prior_mean : bool
                Whether to get the parameters of the prior mean in the solution.
            equal_size : bool
                Whether the clusters are forced to have the same size.
            use_same_prior_mean : bool
                Whether to use the same prior mean for all models.
            seed : int (optional)
                The random seed used to permute the indicies. 
                If seed=None or False or 0, a random seed is not used.
        
        Returns:
            self: The updated object itself.
        """
        if func is not None:
            self.func=func.copy()
            # Set descriptor of the objective function
            self.use_analytic_prefactor=func.use_analytic_prefactor
            self.use_optimized_noise=func.use_optimized_noise
        if get_prior_mean is not None:
            self.get_prior_mean=get_prior_mean
        if batch_size is not None:
            self.batch_size=int(batch_size)
        if equal_size is not None:
            self.equal_size=equal_size
        if use_same_prior_mean is not None:
            self.use_same_prior_mean=use_same_prior_mean
        if seed is not None:
            self.seed=seed
        # Update the objective function
        if len(kwargs.keys()):
            self.func.update_arguments(**kwargs)
        # Always reset the solution when the objective function is changed 
        self.reset_solution()
        return self
    
    def update_solution(self,fun,theta,hp,model,jac=False,deriv=None,**kwargs):
        if fun<self.sol['fun']:
            self.sol['fun']=fun
            self.sol['x']=np.concatenate([hp[para] for para in sorted(hp.keys())])
            self.sol['hp']=hp.copy()
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=self.get_prior_parameters(model)
        return self.sol
    
    def extract_hp_sol(self,hp,n_full_data,n_batch,**kwargs):
        " Extract the hyperparameter solution from the objective function"
        if self.use_analytic_prefactor or self.use_optimized_noise:
            sol=self.func.get_stored_solution()
            weight=n_batch/n_full_data
            if self.use_analytic_prefactor:
                if 'prefactor' in hp.keys():
                    hp['prefactor']+=sol['hp']['prefactor']*weight
                else:
                    hp['prefactor']=sol['hp']['prefactor']*weight
            if self.use_optimized_noise:
                if 'noise' in hp.keys():
                    hp['noise']+=sol['hp']['noise']*weight
                else:
                    hp['noise']=sol['hp']['noise']*weight
        return hp
    
    def set_same_prior_mean(self,model,features,targets,**kwargs):
        " Set the same prior mean constant for the models. "
        if self.use_same_prior_mean:
            model.prior.update(features,targets,**kwargs)
            prior_parameters=model.prior.get_parameters()
            model.update_arguments(prior=Prior_constant(**prior_parameters))
        return model
    
    def get_number_batches(self,n_data,**kwargs):
        " Calculate the number of batches. "
        n_batches=int(n_data//self.batch_size)
        if n_data-(n_batches*self.batch_size):
            n_batches=n_batches+1
        return n_batches

    def randomized_batches(self,indicies,n_data,n_batches,**kwargs):
        " Randomized indicies used for batches. "
        # Permute the indicies
        i_perm=self.get_permutation(indicies)
        # Ensure equal sizes of batches if chosen
        if self.equal_size:
            i_perm=self.ensure_equal_sizes(i_perm,n_data,n_batches)
        i_batches=np.array_split(i_perm,n_batches)
        return i_batches
    
    def get_permutation(self,indicies):
        " Permute the indicies "
        if self.seed:
            rng=np.random.default_rng(seed=self.seed)
            return rng.permutation(indicies)
        return np.random.permutation(indicies)
    
    def ensure_equal_sizes(self,i_perm,n_data,n_batches,**kwargs):
        " Extend the permuted indicies so the clusters have equal sizes. "
        # Find the number of points that should be added
        n_missing=(n_batches*self.batch_size)-n_data
        # Extend the permuted indicies
        if n_missing>0:
            if n_missing>n_data:
                i_perm=np.append(i_perm,np.tile(i_perm,(n_missing//n_data)+1)[:n_missing])
            else:
                i_perm=np.append(i_perm,i_perm[:n_missing])
        return i_perm
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(func=self.func,
                        get_prior_mean=self.get_prior_mean,
                        batch_size=self.batch_size,
                        equal_size=self.equal_size,
                        use_same_prior_mean=self.use_same_prior_mean,
                        seed=self.seed)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    