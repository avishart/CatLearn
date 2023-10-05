from scipy.optimize import OptimizeResult
import numpy as np

class Optimizer:
    def __init__(self,maxiter=5000,jac=True,**kwargs):
        """
        The optimizer used for optimzing the objective function wrt. the hyperparameters.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
        """
        # This optimizer can not be parallelized
        self.parallel=False
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,jac=jac,**kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        """
        Run the optimization method by optimizing the objective function wrt. the hyperparameters.
        Parameters:
            func : ObjectiveFunction class object
                The objective function class that is used to calculate the value.
            theta : (H) array
                An array with the hyperparameter values.
            parameters : (H) list of strings
                A list of names of the hyperparameters.
            model : Model class object
                The Machine Learning Model with kernel and prior that are optimized.
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
        Returns:
            dict : A solution dictionary with objective function value, optimized hyperparameters,
                success statement, and number of used evaluations.
        """
        raise NotImplementedError()

    def update_arguments(self,maxiter=None,jac=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
        Returns:
            self: The updated object itself.
        """
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if jac is not None:
            self.jac=jac
        return self
    
    def get_final_solution(self,sol,func,parameters,model,X,Y,pdis,**kwargs):
        " Get the final solution from the objective function. "
        if self.parallel:
            sol=self.get_final_solution_parallel(sol,func,parameters,model,X,Y,pdis,**kwargs)
        else:
            sol=func.get_solution(sol,parameters,model,X,Y,pdis)
        return OptimizeResult(**sol)
    
    def get_final_solution_parallel(self,sol,func,parameters,model,X,Y,pdis,**kwargs):
        " Get all final solutions from each function at each rank. "
        from ase.parallel import world,broadcast
        size=world.size
        fun_sol=func.get_stored_solution()
        sol=func.get_solution(sol,parameters,model,X,Y,pdis)
        fun_sols=[broadcast(fun_sol['fun'],root=r) for r in range(size)]
        rank_min=np.argmin(fun_sols)
        return broadcast(sol,root=rank_min)
    
    def get_empty_solution(self,**kwargs):
        " Get an empty solution without any function evaluations. "
        sol={'fun':np.inf,'x':np.array([]),
             'success':False,'nfev':0,'nit':0,
             'message':"No function value calculated."}
        return sol
    
    def get_initial_solution(self,theta,func,func_args=(),**kwargs):
        " Get a solution with the evaluation of the initial hyperparameters. "
        sol={'fun':np.inf,'x':theta,
             'success':False,'nfev':1,'nit':1,
             'message':"Function value is calculated."}
        fun=self.get_fun(func)
        if self.jac:
            sol['fun'],sol['jac']=fun(theta,*func_args)
        else:
            sol['fun']=fun(theta,*func_args)
        return sol
    
    def get_func_arguments(self,parameters,model,X,Y,pdis,jac,**kwargs):
        " Get the arguments needed for the objective function. "
        return (parameters,model,X,Y,pdis,jac)
    
    def get_fun(self,func,**kwargs):
        " Get the function that evaluates the objective function. "
        return func.function
    
    def calculate_values(self,thetas,func,func_args=(),**kwargs):
        " Calculate a list of values with a function. "
        if self.parallel:
            return self.calculate_values_parallel(thetas,func,func_args=func_args,**kwargs)
        return np.array([func.function(theta,*func_args) for theta in thetas])
    
    def calculate_values_parallel(self,thetas,func,func_args=(),**kwargs):
        " Calculate a list of values with a function in parallel. "
        from ase.parallel import world,broadcast
        rank,size=world.rank,world.size
        f_list=np.array([func.function(theta,*func_args) for t,theta in enumerate(thetas) if rank==t%size])
        return np.array([broadcast(f_list,root=r) for r in range(size)]).T.reshape(-1)
    
    def compare_solutions(self,sol1,sol2,**kwargs):
        " Compare two solutions and use the solution with lowest function value. "
        # Store the number of used iterations
        nfev=sol1['nfev']+sol2['nfev']
        nit=sol1['nit']+sol2['nit']
        # Compare function values
        if sol2['fun']<sol1['fun']:
            # Solution 2 is best
            sol2['nfev']=nfev
            sol2['nit']=nit
            return sol2
        # Solution 1 is best
        sol1['nfev']=nfev
        sol1['nit']=nit
        return sol1

    def make_hp(self,theta,parameters,**kwargs):
        " Make hyperparameter dictionary from lists. "
        theta,parameters=np.array(theta),np.array(parameters)
        parameters_set=sorted(set(parameters))
        hp={para_s:theta[parameters==para_s] for para_s in parameters_set}
        return hp

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,jac=self.jac)
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


class FunctionEvaluation(Optimizer):
    def __init__(self,jac=True,**kwargs):
        """
        A method used for evaluating the objective function for the given hyperparameters.
        Parameters:
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
        """
        # This optimizer can not be parallelized
        self.parallel=False
        # Set all the arguments
        self.update_arguments(jac=jac,**kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        """
        Run the evaluation of the objective function wrt. the hyperparameters.
        Parameters:
            func : ObjectiveFunction class object
                The objective function class that is used to calculate the value.
            theta : (H) array
                An array with the hyperparameter values.
            parameters : (H) list of strings
                A list of names of the hyperparameters.
            model : Model class object
                The Machine Learning Model with kernel and prior that are optimized.
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
        Returns:
            dict : A solution dictionary with objective function value, hyperparameters,
                success statement, and number of used evaluations.
        """
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,jac=self.jac,**kwargs)
        sol=self.get_initial_solution(theta,func,func_args=func_args)
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)

    def update_arguments(self,maxiter=None,jac=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
        Returns:
            self: The updated object itself.
        """
        if jac is not None:
            self.jac=jac
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(jac=self.jac)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    
