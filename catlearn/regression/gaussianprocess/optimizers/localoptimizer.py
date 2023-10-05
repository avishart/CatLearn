from .optimizer import Optimizer
import numpy as np

class LocalOptimizer(Optimizer):
    def __init__(self,maxiter=5000,jac=True,tol=1e-3,**kwargs):
        """
        The local optimizer used for optimzing the objective function wrt. the hyperparameters.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            tol : float
                A tolerance criterion for convergence. 
        """
        # This optimizer can not be parallelized
        self.parallel=False
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,
                              jac=jac,
                              tol=tol,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        raise NotImplementedError()

    def update_arguments(self,maxiter=None,jac=None,tol=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            tol : float
                A tolerance criterion for convergence. 
        Returns:
            self: The updated object itself.
        """
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if jac is not None:
            self.jac=jac
        if tol is not None:
            self.tol=tol
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,jac=self.jac,tol=self.tol)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    

class ScipyOptimizer(LocalOptimizer):
    def __init__(self,maxiter=5000,jac=True,tol=1e-8,method='l-bfgs-b',bounds=None,use_bounds=False,options={},opt_kwargs={},**kwargs):
        """
        The local optimizer used for optimzing the objective function wrt. the hyperparameters.
        This method uses the SciPy minimizers. 
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            tol : float
                A tolerance criterion for convergence. 
            method : str
                The minimizer method used in SciPy.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                All global optimization methods are using boundary conditions. 
            use_bounds : bool
                Whether to use the boundary conditions or not.
                Only some methods can use boundary conditions.
            options : dict
                Solver options used in the SciPy minimizer.
            opt_kwargs : dict
                Extra arguments used in the SciPy minimizer.
        """
        # This optimizer can not be parallelized
        self.parallel=False
        # Set boundary conditions
        self.bounds=None
        # Set options
        self.options={}
        # Set optimization arguments
        self.opt_kwargs={}
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,
                              jac=jac,
                              tol=tol,
                              method=method,
                              bounds=bounds,
                              use_bounds=use_bounds,
                              options=options,
                              opt_kwargs=opt_kwargs,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        from scipy.optimize import minimize
        # Get the objective function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,self.jac)
        # Get bounds or set it to default argument
        if self.use_bounds:
            bounds=self.make_bounds(parameters,array=True)
        else:
            bounds=None
        # Minimize objective function with SciPy
        sol=minimize(self.get_fun(func),x0=theta,
                     method=self.method,jac=self.jac,
                     tol=self.tol,args=func_args,
                     bounds=bounds,
                     options=self.options,**self.opt_kwargs)
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)

    def update_arguments(self,maxiter=None,jac=None,tol=None,method=None,bounds=None,use_bounds=None,options=None,opt_kwargs=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            tol : float
                A tolerance criterion for convergence. 
            method : str
                The minimizer method used in SciPy.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                All global optimization methods are using boundary conditions. 
            use_bounds : bool
                Whether to use the boundary conditions or not.
                Only some methods can use boundary conditions.
            options : dict
                Solver options used in the SciPy minimizer.
            opt_kwargs : dict
                Extra arguments used in the SciPy minimizer.
        Returns:
            self: The updated object itself.
        """
        if jac is not None:
            self.jac=jac
        if tol is not None:
            self.tol=tol
        if method is not None:
            self.method=method.lower()
            # If method is updated then maxiter must be updated
            if maxiter is None:
                maxiter=self.maxiter
        if options is not None:
            self.options.update(options)
        if maxiter is not None:
            self.maxiter=int(maxiter)
            if self.method in ['nelder-mead']:
                self.options['maxfev']=self.maxiter
            elif self.method in ['l-bfgs-b','tnc']:
                self.options['maxfun']=self.maxiter
            else:
                self.options['maxiter']=self.maxiter
        if bounds is not None:
            self.bounds=bounds.copy()
        if use_bounds is not None:
            if self.bounds is not None and self.method in ['nelder-mead','l-bfgs-b','tnc','slsqp','powell','trust-constr','cobyla']:
                self.use_bounds=use_bounds
            else:
                self.use_bounds=False
        if opt_kwargs is not None:
            self.opt_kwargs.update(opt_kwargs)
        return self
    
    def make_bounds(self,parameters,array=True,**kwargs):
        " Make the boundary conditions of the hyperparameters. "
        return self.bounds.get_bounds(parameters=parameters,array=array,**kwargs)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,jac=self.jac,tol=self.tol,
                        method=self.method,bounds=self.bounds,
                        use_bounds=self.use_bounds,options=self.options,
                        opt_kwargs=self.opt_kwargs)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs


class ScipyPriorOptimizer(ScipyOptimizer):
    def __init__(self,maxiter=5000,jac=True,tol=1e-8,method='l-bfgs-b',bounds=None,use_bounds=False,options={},opt_kwargs={},**kwargs):
        """
        The local optimizer used for optimzing the objective function wrt. the hyperparameters.
        This method uses the SciPy minimizers. 
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        If prior distributions of the hyperparameters are used, it will start by include
        the prior distributions and then restart with excluded prior distributions.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            tol : float
                A tolerance criterion for convergence. 
            method : str
                The minimizer method used in SciPy.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                All global optimization methods are using boundary conditions. 
            use_bounds : bool
                Whether to use the boundary conditions or not.
                Only some methods can use boundary conditions.
            options : dict
                Solver options used in the SciPy minimizer.
            opt_kwargs : dict
                Extra arguments used in the SciPy minimizer.
        """
        super().__init__(maxiter=maxiter,jac=jac,tol=tol,method=method,
                         bounds=bounds,use_bounds=use_bounds,options=options,
                         opt_kwargs=opt_kwargs,**kwargs)
        
    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        # Check if prior distributions of the hyperparameters are used
        if pdis is None:
            return super().run(func,theta,parameters,model,X,Y,pdis,**kwargs)
        # If the prior distributions of the hyperparameters are used then include them in the optimization
        sol=super().run(func,theta,parameters,model,X,Y,pdis,**kwargs)
        # Save the number of evaluations and the new best hyperparameters
        nfev=sol['nfev']
        # Exclude the prior distributions of the hyperparameters in the optimization
        sol=super().run(func,sol['x'],parameters,model,X,Y,None,**kwargs)
        sol['nfev']+=nfev
        sol['nit']=2
        return sol
    

class ScipyGuessOptimizer(ScipyOptimizer):
    def __init__(self,maxiter=5000,jac=True,tol=1e-8,method='l-bfgs-b',bounds=None,use_bounds=False,options={},opt_kwargs={},**kwargs):
        """
        The local optimizer used for optimzing the objective function wrt. the hyperparameters.
        This method uses the SciPy minimizers. 
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        Use boundary conditions to give an extra guess of the hyperparameters that also are optimized.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            tol : float
                A tolerance criterion for convergence. 
            method : str
                The minimizer method used in SciPy.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                All global optimization methods are using boundary conditions. 
            use_bounds : bool
                Whether to use the boundary conditions or not.
                Only some methods can use boundary conditions.
            options : dict
                Solver options used in the SciPy minimizer.
            opt_kwargs : dict
                Extra arguments used in the SciPy minimizer.
        """
        super().__init__(maxiter=maxiter,jac=jac,tol=tol,method=method,
                         bounds=bounds,use_bounds=use_bounds,options=options,
                         opt_kwargs=opt_kwargs,**kwargs)
        
    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        # Check if boundary conditions of the hyperparameters are used
        if self.bounds is None:
            return super().run(func,theta,parameters,model,X,Y,pdis,**kwargs)
        # Use the boundary conditions to give an educated guess of the hyperparmeters
        theta_guess=self.guess_hp(parameters,array=True)
        sol_ed=super().run(func,theta_guess,parameters,model,X,Y,pdis,**kwargs)
        # Optimize the initial hyperparameters
        sol=super().run(func,theta,parameters,model,X,Y,pdis,**kwargs)
        # Update the solution if it is better
        sol=self.compare_solutions(sol,sol_ed)
        sol['nit']=2
        return sol
    
    def guess_hp(self,parameters,array=True,**kwargs):
        " Make a guess of the hyperparameters from the boundary conditions. "
        return self.bounds.get_hp(parameters=parameters,array=array,**kwargs)

