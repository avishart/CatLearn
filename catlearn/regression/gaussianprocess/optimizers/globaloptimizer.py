from .optimizer import Optimizer
import numpy as np

class GlobalOptimizer(Optimizer):
    def __init__(self,local_optimizer=None,bounds=None,maxiter=5000,**kwargs):
        """
        The global optimizer used for optimzing the objective function wrt. the hyperparameters.
        The global optimizer requires a local optimization method and boundary conditions of the hyperparameters.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
        """
        # This global optimizer can not be parallelized
        self.parallel=False
        # The gradient of the objective function is not used by the global optimizer
        self.jac=False
        # Set default bounds
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            bounds=VariableTransformation(bounds=None)
        # Set default local optimizer
        if local_optimizer is None:
            from .localoptimizer import ScipyOptimizer
            local_optimizer=ScipyOptimizer(maxiter=maxiter,bounds=bounds,use_bounds=False)
        # Set all the arguments
        self.update_arguments(local_optimizer=local_optimizer,
                              bounds=bounds,
                              maxiter=maxiter,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        raise NotImplementedError()

    def update_arguments(self,local_optimizer=None,bounds=None,maxiter=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
        Returns:
            self: The updated object itself.
        """
        if local_optimizer is not None:
            self.local_optimizer=local_optimizer.copy()
        if bounds is not None:
            self.bounds=bounds.copy()
            # Use the same boundary conditions in the local optimizer
            self.local_optimizer.update_arguments(bounds=self.bounds)
        if maxiter is not None:
            self.maxiter=int(maxiter)
        return self
    
    def run_local_opt(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        " Run the local optimization. "
        return self.local_optimizer.run(func,theta,parameters,model,X,Y,pdis,**kwargs)
    
    def make_lines(self,parameters,ngrid,**kwargs):
        " Make the lines of the hyperparameters from boundary conditions. "
        return self.bounds.make_lines(parameters=parameters,ngrid=ngrid,**kwargs)
    
    def make_bounds(self,parameters,array=True,**kwargs):
        " Make the boundary conditions of the hyperparameters. "
        return self.bounds.get_bounds(parameters=parameters,array=array,**kwargs)
    
    def sample_thetas(self,parameters,npoints,**kwargs):
        " Draw random hyperparameter samples from the boundary conditions. "
        return self.bounds.sample_thetas(parameters=parameters,npoints=npoints,**kwargs)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(local_optimizer=self.local_optimizer,
                        bounds=self.bounds,
                        maxiter=self.maxiter)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    

class RandomSamplingOptimizer(GlobalOptimizer):
    def __init__(self,local_optimizer=None,bounds=None,maxiter=5000,npoints=40,parallel=False,**kwargs):
        """
        The random sampling optimizer used for optimzing the objective function wrt. the hyperparameters.
        The random sampling optimizer samples the hyperparameters randomly from the boundary conditions
        and optimize all samples with the local optimizer.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            npoints : int
                The number of hyperparameter points samled from the boundary conditions. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        """
        # The gradient of the objective function is not used by the global optimizer
        self.jac=False
        # Set default bounds
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            bounds=VariableTransformation(bounds=None)
        # Set default local optimizer
        if local_optimizer is None:
            from .localoptimizer import ScipyOptimizer
            local_optimizer=ScipyOptimizer(maxiter=int(maxiter/npoints),bounds=bounds,use_bounds=False)
        # Set all the arguments
        self.update_arguments(local_optimizer=local_optimizer,
                              bounds=bounds,
                              maxiter=maxiter,
                              npoints=npoints,
                              parallel=parallel,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        # Draw random hyperparameter samples
        thetas=np.array([theta])
        if self.npoints>1:
            thetas=self.sample_thetas(parameters,npoints=int(self.npoints-1))
            thetas=np.append(thetas,thetas,axis=0)
        # Make empty solution and lists
        sol=self.get_empty_solution()
        # Perform the local optimization for random samples
        sol=self.optimize_samples(sol,func,thetas,parameters,model,X,Y,pdis,**kwargs)
        return sol
    
    def update_arguments(self,local_optimizer=None,bounds=None,maxiter=None,npoints=None,parallel=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            npoints : int
                The number of hyperparameter points samled from the boundary conditions. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        Returns:
            self: The updated object itself.
        """
        if local_optimizer is not None:
            self.local_optimizer=local_optimizer.copy()
        if bounds is not None:
            self.bounds=bounds.copy()
            # Use the same boundary conditions in the local optimizer
            self.local_optimizer.update_arguments(bounds=self.bounds)
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if parallel is not None:
            self.parallel=parallel
        if npoints is not None:
            if self.parallel:
                from ase.parallel import world
                # Make sure that the optimal number of points is used for the CPUs
                self.npoints=int(int(npoints/world.size)*world.size)
                if self.npoints==0:
                    self.npoints=world.size
            else:
                self.npoints=int(npoints)
        return self
    
    def optimize_samples(self,sol,func,thetas,parameters,model,X,Y,pdis,**kwargs):
        " Perform the local optimization of the random samples. "
        # Check if the optimization should be performed in parallel
        if self.parallel:
            return self.optimize_samples_parallel(sol,func,thetas,parameters,model,X,Y,pdis,**kwargs)
        for theta in thetas:
            # Check if the maximum number of iterations is used
            if sol['nfev']>=self.maxiter:
                break
            # Do local optimization
            sol_s=self.run_local_opt(func,theta,parameters,model,X,Y,pdis,**kwargs)
            # Update the solution if it is better
            sol=self.compare_solutions(sol,sol_s)
        # Update the total number of iterations
        sol['nit']=len(thetas)
        # Get the all best time best solution 
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
    
    def optimize_samples_parallel(self,sol,func,thetas,parameters,model,X,Y,pdis,**kwargs):
        " Perform the local optimization of the random samples in parallel. "
        from ase.parallel import world
        rank,size=world.rank,world.size
        for t,theta in enumerate(thetas):
            if rank==t%size:
                # Check if the maximum number of iterations is used
                if sol['nfev']>=self.maxiter:
                    break
                # Do local optimization in parallel
                sol_s=self.run_local_opt(func,theta,parameters,model,X,Y,pdis,**kwargs)
                # Update the solution if it is better
                sol=self.compare_solutions(sol,sol_s)
        # Update the total number of iterations
        sol['nit']=len(thetas)
        # Get the all best time best solution for all CPUs and broadcast it
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(local_optimizer=self.local_optimizer,
                        bounds=self.bounds,
                        maxiter=self.maxiter,
                        npoints=self.npoints)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs


class GridOptimizer(GlobalOptimizer):
    def __init__(self,local_optimizer=None,bounds=None,maxiter=5000,n_each_dim=None,optimize=True,parallel=False,**kwargs):
        """
        The grid optimizer used for optimzing the objective function wrt. the hyperparameters.
        The grid optimizer makes a grid in the hyperparameter space from the boundary conditions and evaluate them.
        The grid point with the lowest function value can be optimized with the local optimizer.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            n_each_dim : int or (H) list
                An integer or a list with number of grid points in each dimension of the hyperparameters.
            optimize : bool
                Whether to perform a local optimization on the best found solution. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        """
        # The gradient of the objective function is not used by the global optimizer
        self.jac=False
        # Set default bounds
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            bounds=VariableTransformation(bounds=None)
        # Set default local optimizer
        if local_optimizer is None:
            from .localoptimizer import ScipyOptimizer
            local_optimizer=ScipyOptimizer(maxiter=maxiter,bounds=bounds,use_bounds=False)
        # Set n_each_dim as default
        self.n_each_dim=None
        # Set all the arguments
        self.update_arguments(local_optimizer=local_optimizer,
                              bounds=bounds,
                              maxiter=maxiter,
                              n_each_dim=n_each_dim,
                              optimize=optimize,
                              parallel=parallel,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        # Number of points per dimension
        n_each_dim=self.get_n_each_dim(len(theta))
        # Make grid either with the same or different numbers in each dimension
        lines=self.make_lines(parameters,ngrid=n_each_dim)
        thetas=np.append([theta],self.make_grid(lines,maxiter=int(self.maxiter-1)),axis=0)
        # Check if the number of points is well parallized if it is used
        thetas=self.check_npoints(thetas)
        # Make empty solution and lists
        sol=self.get_empty_solution()
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,jac=False,**kwargs)
        # Calculate the grid points
        f_list=self.calculate_values(thetas,func,func_args=func_args)
        # Find the minimum function value
        sol=self.get_minimum(sol,thetas,f_list)
        sol=self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
        # Perform the local optimization for the minimum function value
        sol=self.optimize_minimum(sol,func,parameters,model,X,Y,pdis,**kwargs)
        return sol
    
    def update_arguments(self,local_optimizer=None,bounds=None,maxiter=None,n_each_dim=None,optimize=None,parallel=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            n_each_dim : int or (H) list
                An integer or a list with number of grid points in each dimension of the hyperparameters.
            optimize : bool
                Whether to perform a local optimization on the best found solution. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        Returns:
            self: The updated object itself.
        """
        if local_optimizer is not None:
            self.local_optimizer=local_optimizer.copy()
        if bounds is not None:
            self.bounds=bounds.copy()
            # Use the same boundary conditions in the local optimizer
            self.local_optimizer.update_arguments(bounds=self.bounds)
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if parallel is not None:
            self.parallel=parallel
        if n_each_dim is not None:
            if isinstance(n_each_dim,(list,np.ndarray)):
                self.n_each_dim=n_each_dim.copy()
            else:
                self.n_each_dim=n_each_dim
        if optimize is not None:
            self.optimize=optimize
        return self
    
    def make_grid(self,lines,maxiter=5000):
        " Make a grid in multi-dimensions from a list of 1D grids in each dimension. "
        lines=np.array(lines)
        if len(lines.shape)<2:
            lines=lines.reshape(1,-1)
        # Number of combinations
        combi=1
        for i in [len(line) for line in lines]:
            combi*=i
        if combi<maxiter:
            maxiter=combi
        # If there is a low probability to find grid points randomly the entire grid are calculated
        if (1-(maxiter/combi))<0.99:
            X=lines[0].reshape(-1,1)
            lines=lines[1:]
            for line in lines:
                dim_X=len(X)
                X=np.concatenate([X]*len(line),axis=0)
                X=np.concatenate([X,np.sort(np.concatenate([line.reshape(-1)]*dim_X,axis=0)).reshape(-1,1)],axis=1)
            return np.random.permutation(X)[:maxiter]
        # Randomly sample the grid points
        X=np.array([np.random.choice(line,size=maxiter) for line in lines]).T
        X=np.unique(X,axis=0)
        while len(X)<maxiter:
            x=np.array([np.random.choice(line,size=1) for line in lines]).T
            X=np.append(X,x,axis=0)
            X=np.unique(X,axis=0)
        return X[:maxiter]
    
    def optimize_minimum(self,sol,func,parameters,model,X,Y,pdis,**kwargs):
        " Perform the local optimization of the found minimum. "
        # Check if optimization should be used
        if not self.optimize:
            return sol
        # Check if all iterations have been used
        if sol['nfev']>=self.maxiter:
            return sol
        # Perform local optimization
        sol_s=self.run_local_opt(func,sol['x'],parameters,model,X,Y,pdis,**kwargs)
        # Update the solution if it is better
        sol=self.compare_solutions(sol,sol_s)
        # Update the number of used iterations
        sol['nit']+=1
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
    
    def get_n_each_dim(self,dim,**kwargs):
        " Number of points per dimension. "
        if self.n_each_dim is None:
            n_each_dim=int((self.maxiter-1)**(1/dim))
            n_each_dim=n_each_dim if n_each_dim>1 else 1
        else:
            n_each_dim=self.n_each_dim
        return n_each_dim
    
    def check_npoints(self,thetas,**kwargs):
        " Check if the number of points is well parallized if it is used. "
        if self.parallel:
            from ase.parallel import world
            npoints=int(int(len(thetas)/world.size)*world.size)
            if npoints==0:
                npoints=world.size
            return thetas[:npoints]
        return thetas
    
    def get_minimum(self,sol,thetas,f_list,**kwargs):
        " Find the minimum function value and update the solution. "
        # Find the minimum function value
        i_min=np.nanargmin(f_list)
        # Get the number of used iterations
        thetas_len=len(thetas)
        # Update the number of used iterations
        sol['nfev']+=thetas_len
        sol['nit']+=thetas_len
        # Check if a better point is found
        if f_list[i_min]>sol['fun']:
            return sol
        # Update the solution if a better point is found
        sol['fun']=f_list[i_min]
        sol['x']=thetas[i_min].copy()
        sol['message']="Lower function value found."
        return sol
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(local_optimizer=self.local_optimizer,
                        bounds=self.bounds,
                        maxiter=self.maxiter,
                        n_each_dim=self.n_each_dim,
                        optimize=self.optimize,
                        parallel=self.parallel)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs


class IterativeLineOptimizer(GridOptimizer):
    def __init__(self,local_optimizer=None,bounds=None,maxiter=5000,n_each_dim=None,loops=3,calculate_init=False,optimize=True,parallel=False,**kwargs):
        """
        The iteratively line optimizer used for optimzing the objective function wrt. the hyperparameters.
        The iteratively line optimizer makes a 1D grid in each dimension of the hyperparameter space from the boundary conditions.
        The grid points are then evaluated and the best value updates the hyperparameter in the specific dimension.
        This process is done iteratively over all dimensions and in loops. 
        The grid point with the lowest function value can be optimized with the local optimizer.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            n_each_dim : int or (H) list
                An integer or a list with number of grid points in each dimension of the hyperparameters.
            loops : int
                The number of times all the hyperparameter dimensions have been searched.
            calculate_init : bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point. 
            optimize : bool
                Whether to perform a local optimization on the best found solution. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        """
        super().__init__(local_optimizer=local_optimizer,
                         bounds=bounds,
                         maxiter=maxiter,
                         n_each_dim=n_each_dim,
                         loops=loops,
                         calculate_init=calculate_init,
                         optimize=optimize,
                         parallel=parallel,
                         **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        # Number of points per dimension
        n_each_dim=self.get_n_each_dim(len(theta))
        # Make grid either with the same or different numbers in each dimension
        lines=self.make_lines(parameters,ngrid=n_each_dim)
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,jac=False,**kwargs)
        # Calculate the grid points in the iterative grid/line search
        sol=self.iterative_line(theta,lines,func,func_args=func_args)
        sol=self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
        # Perform the local optimization for the minimum function value
        sol=self.optimize_minimum(sol,func,parameters,model,X,Y,pdis,**kwargs)
        return sol
    
    def update_arguments(self,local_optimizer=None,bounds=None,maxiter=None,n_each_dim=None,loops=None,calculate_init=None,optimize=None,parallel=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            local_optimizer : Local optimizer class
                A local optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            n_each_dim : int or (H) list
                An integer or a list with number of grid points in each dimension of the hyperparameters.
            loops : int
                The number of times all the hyperparameter dimensions have been searched.
            calculate_init : bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point. 
            optimize : bool
                Whether to perform a local optimization on the best found solution. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        Returns:
            self: The updated object itself.
        """
        if local_optimizer is not None:
            self.local_optimizer=local_optimizer.copy()
        if bounds is not None:
            self.bounds=bounds.copy()
            # Use the same boundary conditions in the local optimizer
            self.local_optimizer.update_arguments(bounds=self.bounds)
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if parallel is not None:
            self.parallel=parallel
        if loops is not None:
            self.loops=int(loops)
        if calculate_init is not None:
            self.calculate_init=calculate_init
        if n_each_dim is not None:
            if isinstance(n_each_dim,(list,np.ndarray)):
                if np.sum(n_each_dim)*self.loops>self.maxiter:
                    self.n_each_dim=self.get_n_each_dim(len(n_each_dim))
                else:
                    self.n_each_dim=n_each_dim.copy()
            else:
                self.n_each_dim=n_each_dim
        if optimize is not None:
            self.optimize=optimize
        return self
    
    def iterative_line(self,theta,lines,func,func_args=(),**kwargs):
        " Perform iteratively grid/line search. "
        # Make an initial solution
        if self.calculate_init:
            sol=self.get_initial_solution(theta,func,func_args=func_args)
        else:
            sol=self.get_empty_solution()
        # Get the dimension list
        dims=list(range(len(lines)))
        # Set initidal dimension
        d=None
        # Perform loops
        for l in range(self.loops):
            # Permute the dimensions
            dim_perm=np.random.permutation(dims)
            # Make sure the same dimension is not used after each other
            if dim_perm[0]==d:
                dim_perm=dim_perm[1:]
            for d in dim_perm:
                # Make the hyperparameter changes to the specific dimension
                thetas=np.tile(theta,(len(lines[d]),1))
                thetas[:,d]=lines[d].copy()
                f_list=self.calculate_values(thetas,func,func_args=func_args)
                sol=self.get_minimum(sol,thetas,f_list)
                theta=sol['x'].copy()
        return sol
    
    def get_n_each_dim(self,dim):
        " Number of points per dimension. "
        if self.n_each_dim is None:
            n_each_dim=int(self.maxiter/(self.loops*dim))
            n_each_dim=n_each_dim if n_each_dim>1 else 1
        else:
            n_each_dim=self.n_each_dim
        if self.parallel:
            return self.get_n_each_dim_parallel(n_each_dim)
        return n_each_dim
    
    def get_n_each_dim_parallel(self,n_each_dim):
        " Number of points per dimension if it is parallelized. "
        from ase.parallel import world
        if isinstance(n_each_dim,(list,np.ndarray)):
            for d,n_dim in enumerate(n_each_dim):
                n_each_dim[d]=int(int(n_dim/world.size)*world.size)
                if n_each_dim[d]==0:
                    n_each_dim[d]=world.size
        else:
            n_each_dim=int(int(n_each_dim/world.size)*world.size)
            if n_each_dim==0:
                n_each_dim=world.size
        return n_each_dim
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(local_optimizer=self.local_optimizer,
                        bounds=self.bounds,
                        maxiter=self.maxiter,
                        n_each_dim=self.n_each_dim,
                        loops=self.loops,
                        calculate_init=self.calculate_init,
                        optimize=self.optimize,
                        parallel=self.parallel)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs


class BasinOptimizer(GlobalOptimizer):
    def __init__(self,maxiter=5000,jac=True,opt_kwargs={},local_kwargs={},**kwargs):
        """
        The basin-hopping optimizer used for optimzing the objective function wrt. the hyperparameters.
        The basin-hopping optimizer is a wrapper to SciPy's basinhopping.
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
        No local optimizer and boundary conditions are given to this optimizer. 
        The local optimizer is set by keywords in the local_kwargs and it uses SciPy's minimizer.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            opt_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's basinhopping.
            local_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's local minimizer.
        """
        # This global optimizer can not be parallelized
        self.parallel=False
        # Set default arguments for SciPy's basinhopping
        self.opt_kwargs=dict(niter=5,interval=10,T=1.0,stepsize=0.1,niter_success=None)
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs=dict(options={'maxiter':int(maxiter/self.opt_kwargs['niter'])})
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,
                              jac=jac,
                              opt_kwargs=opt_kwargs,
                              local_kwargs=local_kwargs,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        from scipy.optimize import basinhopping
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,self.jac,**kwargs)
        # Get the function that evaluate the objective function
        fun=self.get_fun(func)
        # Set the minimizer kwargs
        minimizer_kwargs=dict(args=func_args,jac=self.jac,**self.local_kwargs)
        # Do the basin-hopping
        sol=basinhopping(fun,x0=theta,minimizer_kwargs=minimizer_kwargs,**self.opt_kwargs)
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
    
    def update_arguments(self,maxiter=None,jac=None,opt_kwargs=None,local_kwargs=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            opt_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's basinhopping.
            local_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's local minimizer.
        Returns:
            self: The updated object itself.
        """
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if jac is not None:
            self.jac=jac
        if opt_kwargs is not None:
            self.opt_kwargs.update(opt_kwargs)
        if local_kwargs is not None:
            if 'options' in local_kwargs:
                local_no_options={key:value for key,value in local_kwargs.items() if key!='options'}
                self.local_kwargs.update(local_no_options)
                self.local_kwargs['options'].update(local_kwargs['options'])
            else:
                self.local_kwargs.update(local_kwargs)
        # Make sure not to many iterations are used in average
        maxiter_niter=int(self.maxiter/self.opt_kwargs['niter'])
        if maxiter_niter<self.local_kwargs['options']['maxiter']:
            self.local_kwargs['options']['maxiter']=maxiter_niter
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,jac=self.jac,
                        opt_kwargs=self.opt_kwargs,
                        local_kwargs=self.local_kwargs)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    

class AnneallingOptimizer(GlobalOptimizer):
    def __init__(self,bounds=None,maxiter=5000,jac=True,opt_kwargs={},local_kwargs={},**kwargs):
        """
        The simulated annealing optimizer used for optimzing the objective function wrt. the hyperparameters.
        The simulated annealing optimizer is a wrapper to SciPy's dual_annealing.
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
        No local optimizer is given to this optimizer. 
        The local optimizer is set by keywords in the local_kwargs and it uses SciPy's minimizer.
        Parameters:
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            opt_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's dual_annealing.
            local_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's local minimizer.
        """
        # Set default bounds
        if bounds is None:
            from ..hpboundary.educated import EducatedBoundaries
            bounds=EducatedBoundaries(log=True)
        # This global optimizer can not be parallelized
        self.parallel=False
        # Set default arguments for SciPy's dual_annealing
        self.opt_kwargs=dict(initial_temp=5230.0,restart_temp_ratio=2e-05,
                             visit=2.62,accept=-5.0,seed=None,no_local_search=False)
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs=dict(options={})
        # Set all the arguments
        self.update_arguments(bounds=bounds,
                              maxiter=maxiter,
                              jac=jac,
                              opt_kwargs=opt_kwargs,
                              local_kwargs=local_kwargs,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        from scipy.optimize import dual_annealing
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,jac=False,**kwargs)
        # Get the function
        fun=self.get_fun(func)
        # Set the minimizer kwargs
        minimizer_kwargs=dict(jac=False,**self.local_kwargs)
        # Make boundary conditions
        bounds=self.make_bounds(parameters,array=True)
        # Do the dual simulated annealing
        sol=dual_annealing(fun,bounds=bounds,
                           x0=theta,args=func_args,
                           maxiter=self.maxiter,maxfun=self.maxiter,
                           minimizer_kwargs=minimizer_kwargs,**self.opt_kwargs)        
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
    
    def update_arguments(self,bounds=None,maxiter=None,jac=None,opt_kwargs=None,local_kwargs=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            opt_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's dual_annealing.
            local_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's local minimizer.
        Returns:
            self: The updated object itself.
        """
        if bounds is not None:
            self.bounds=bounds.copy()
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if jac is not None:
            self.jac=jac
        if opt_kwargs is not None:
            self.opt_kwargs.update(opt_kwargs)
        if local_kwargs is not None:
            if 'options' in local_kwargs:
                local_no_options={key:value for key,value in local_kwargs.items() if key!='options'}
                self.local_kwargs.update(local_no_options)
                self.local_kwargs['options'].update(local_kwargs['options'])
            else:
                self.local_kwargs.update(local_kwargs)
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(bounds=self.bounds,maxiter=self.maxiter,
                        jac=self.jac,opt_kwargs=self.opt_kwargs,
                        local_kwargs=self.local_kwargs)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs


class AnneallingTransOptimizer(AnneallingOptimizer):
    def __init__(self,bounds=None,maxiter=5000,jac=True,opt_kwargs={},local_kwargs={},**kwargs):
        """
        The simulated annealing optimizer used for optimzing the objective function wrt. the hyperparameters.
        The simulated annealing optimizer is a wrapper to SciPy's dual_annealing.
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
        No local optimizer is given to this optimizer. 
        The local optimizer is set by keywords in the local_kwargs and it uses SciPy's minimizer.
        This simulated annealing optimizer uses variable transformation of the hyperparameters to search the space.
        Parameters:
            bounds : VariableTransformation class
                A class of the variable transformation of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            opt_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's dual_annealing.
            local_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's local minimizer.
        """
        # Set default bounds
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            bounds=VariableTransformation(bounds=None)
        # This global optimizer can not be parallelized
        self.parallel=False
        # Set default arguments for SciPy's dual_annealing
        self.opt_kwargs=dict(initial_temp=5230.0,restart_temp_ratio=2e-05,
                             visit=2.62,accept=-5.0,seed=None,no_local_search=False)
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs=dict(options={})
        # Set all the arguments
        self.update_arguments(bounds=bounds,
                              maxiter=maxiter,
                              jac=jac,
                              opt_kwargs=opt_kwargs,
                              local_kwargs=local_kwargs,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        from scipy.optimize import dual_annealing
        # Get the function arguments for the wrappers
        func_args_w=self.get_wrapper_arguments(func,parameters,model,X,Y,pdis,jac=self.jac,**kwargs)
        # Set the minimizer kwargs
        minimizer_kwargs=dict(jac=False,**self.local_kwargs)
        # Make boundary conditions
        bounds=self.make_bounds(parameters,array=True,transformed=True)
        # Do the dual simulated annealing
        sol=dual_annealing(self.func_vartrans,bounds=bounds,
                           x0=theta,args=func_args_w,
                           maxiter=self.maxiter,maxfun=self.maxiter,
                           minimizer_kwargs=minimizer_kwargs,**self.opt_kwargs)
        sol=self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
        # Retransform hyperparameters
        sol=self.transform_solution(sol)
        return sol
    
    def update_arguments(self,bounds=None,maxiter=None,jac=None,opt_kwargs=None,local_kwargs=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            bounds : VariableTransformation class
                A class of the variable transformation of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            jac : bool
                Whether to use the gradient of the objective function wrt. the hyperparameters.
            opt_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's dual_annealing.
            local_kwargs : dict
                A dictionary with the arguments and keywords given to SciPy's local minimizer.
        Returns:
            self: The updated object itself.
        """
        if bounds is not None:
            from ..hpboundary.hptrans import VariableTransformation
            if not isinstance(bounds,VariableTransformation):
                raise Exception('A variable transformation as bounds has to be used!')
            self.bounds=bounds.copy()
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if jac is not None:
            self.jac=jac
        if opt_kwargs is not None:
            self.opt_kwargs.update(opt_kwargs)
        if local_kwargs is not None:
            if 'options' in local_kwargs:
                local_no_options={key:value for key,value in local_kwargs.items() if key!='options'}
                self.local_kwargs.update(local_no_options)
                self.local_kwargs['options'].update(local_kwargs['options'])
            else:
                self.local_kwargs.update(local_kwargs)
        return self

    def func_vartrans(self,ti,fun,parameters,func_args=(),**kwargs):
        " Objective function called for simulated annealing, where hyperparameters are transformed. "
        theta=self.reverse_trasformation(ti,parameters)
        return fun(theta,*func_args)
    
    def reverse_trasformation(self,ti,parameters,**kwargs):
        " Transform the variable transformed hyperparameters back to hyperparameter log-space. "
        ti=np.where(ti<1.0,np.where(ti>0.0,ti,self.bounds.eps),1.00-self.bounds.eps)
        t=self.make_hp(ti,parameters)
        theta=self.bounds.reverse_trasformation(t,array=True)
        return theta
    
    def transform_solution(self,sol,**kwargs):
        " Retransform the variable transformed hyperparameters in the solution back to hyperparameter log-space. "
        sol['x']=self.bounds.reverse_trasformation(sol['hp'],array=True)
        sol['hp']=self.bounds.reverse_trasformation(sol['hp'],array=False)
        return sol
    
    def get_wrapper_arguments(self,func,parameters,model,X,Y,pdis,jac,**kwargs):
        " Get the function arguments for the wrappers. "
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,jac=False,**kwargs)
        # Get the function that evaluate the objective function
        fun=self.get_fun(func)
        # Get the function arguments for the wrappers
        func_args_w=(fun,parameters,func_args)
        return func_args_w


class FactorizedOptimizer(GlobalOptimizer):
    def __init__(self,line_optimizer=None,bounds=None,maxiter=5000,ngrid=80,calculate_init=False,parallel=False,**kwargs):
        """
        The factorized optimizer used for optimzing the objective function wrt. the hyperparameters.
        The factorized optimizer makes a 1D grid for each hyperparameter from the boundary conditions. 
        The hyperparameters are then optimized with a line search optimizer. 
        The line search optimizer optimizes only one of the hyperparameters and 
        it therefore relies on a factorization method as the objective function. 
        Parameters:
            line_optimizer : Line search optimizer class
                A line search optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            ngrid : int
                The number of grid points of the hyperparameter that is optimized.
            calculate_init : bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        """
        # The gradient of the objective function is not used by the global optimizer
        self.jac=False
        # Set default bounds
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            bounds=VariableTransformation(bounds=None)
        # Set default line optimizer
        if line_optimizer is None:
            from .linesearcher import GoldenSearch
            line_optimizer=GoldenSearch(maxiter=int(maxiter),parallel=parallel)
        # Set all the arguments
        self.update_arguments(line_optimizer=line_optimizer,
                              bounds=bounds,
                              maxiter=maxiter,
                              ngrid=ngrid,
                              calculate_init=calculate_init,
                              parallel=parallel,
                              **kwargs)

    def run(self,func,theta,parameters,model,X,Y,pdis,**kwargs):
        # Make an initial solution or use an empty solution
        if self.calculate_init:
            func_args=self.get_func_arguments(parameters,model,X,Y,pdis,jac=False,**kwargs)
            sol=self.get_initial_solution(theta,func,func_args=func_args)
        else:
            sol=self.get_empty_solution()
        # Make the lines of the hyperparameters
        lines=np.array(self.make_lines(parameters,ngrid=self.ngrid)).T
        # Optimize the hyperparameters with the line search
        sol_s=self.run_line_opt(func,lines,parameters,model,X,Y,pdis,**kwargs)
        # Update the solution if it is better
        sol=self.compare_solutions(sol,sol_s)
        # Change the solution message
        if sol['success']:
            sol['message']='Local optimization is converged.'
        else:
            sol['message']='Local optimization is not converged.'
        return self.get_final_solution(sol,func,parameters,model,X,Y,pdis)
    
    def update_arguments(self,line_optimizer=None,bounds=None,maxiter=None,ngrid=None,calculate_init=None,parallel=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            line_optimizer : Line search optimizer class
                A line search optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter : int
                The maximum number of evaluations or iterations the global optimizer can use.
            ngrid : int
                The number of grid points of the hyperparameter that is optimized.
            calculate_init : bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point. 
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
        Returns:
            self: The updated object itself.
        """
        if line_optimizer is not None:
            self.line_optimizer=line_optimizer.copy()
        if bounds is not None:
            self.bounds=bounds.copy()
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if parallel is not None:
            self.parallel=parallel
        if ngrid is not None:
            if self.parallel:
                from ase.parallel import world
                self.ngrid=int(int(ngrid/world.size)*world.size)
                if self.ngrid==0:
                    self.ngrid=world.size
            else:
                self.ngrid=ngrid
        if calculate_init is not None:
            self.calculate_init=calculate_init
        return self
    
    def run_line_opt(self,func,lines,parameters,model,X,Y,pdis,**kwargs):
        " Run the line search optimization. "
        return self.line_optimizer.run(func,lines,parameters,model,X,Y,pdis,**kwargs)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(line_optimizer=self.line_optimizer,
                        bounds=self.bounds,
                        maxiter=self.maxiter,
                        ngrid=self.ngrid,
                        calculate_init=self.calculate_init,
                        parallel=self.parallel)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs

