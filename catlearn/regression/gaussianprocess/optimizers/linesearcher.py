from .localoptimizer import LocalOptimizer
import numpy as np

class LineSearchOptimizer(LocalOptimizer):
    def __init__(self,maxiter=5000,tol=1e-5,optimize=True,multiple_min=True,theta_index=None,parallel=False,xtol=None,ftol=None,**kwargs):
        """
        The line search optimizer is used for optimzing the objective function wrt. a single hyperparameter.
        The LineSearchOptimizer does only work together with a GlobalOptimizer that uses line searches (e.g. FactorizedOptimizer). 
        A line of the hyperparameter is required to run the line search.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        """
        # Line search optimizers cannot use gradients of the objective function 
        self.jac=False
        # Set the default theta_index
        self.theta_index=None
        # Set xtol and ftol to the tolerance if they are not given.
        xtol,ftol=self.set_tols(tol,xtol=xtol,ftol=ftol)
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,
                              tol=tol,
                              optimize=optimize,
                              multiple_min=multiple_min,
                              theta_index=theta_index,
                              parallel=parallel,
                              xtol=xtol,
                              ftol=ftol,
                              **kwargs)

    def run(self,func,line,parameters,model,X,Y,pdis,**kwargs):
        """
        Run the line search by optimizing the objective function wrt. the hyperparameter.
        The grid/line of the hyperparameter has to be given.
        Parameters:
            func : ObjectiveFunction class object
                The objective function class that is used to calculate the value.
            line : (ngrid,H) array
                An array with the grid points of the hyperparameters.
                Only one of the hyperparameters is used, which is given by theta_index.
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

    def update_arguments(self,maxiter=None,tol=None,optimize=None,multiple_min=None,theta_index=None,parallel=None,xtol=None,ftol=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        Returns:
            self: The updated object itself.
        """
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if tol is not None:
            self.tol=tol
        if optimize is not None:
            self.optimize=optimize
        if multiple_min is not None:
            self.multiple_min=multiple_min
        if theta_index is not None:
            self.theta_index=int(theta_index)
        if parallel is not None:
            self.parallel=parallel
        if xtol is not None:
            self.xtol=xtol
        if ftol is not None:
            self.ftol=ftol
        return self
    
    def find_minimas(self,xvalues,fvalues,i_min,len_l,theta_index,**kwargs):
        " Find all the local minimums and their indicies or just the global minimum and then check convergence. "
        # Investigate multiple minimums
        if self.multiple_min:
            return self.find_multiple_min(xvalues,fvalues,len_l,theta_index)
        # Investigate the global minimum
        return self.find_single_min(xvalues,fvalues,i_min,len_l,theta_index)
    
    def find_multiple_min(self,xvalues,fvalues,len_l,theta_index,**kwargs):
        " Find all the local minimums and their indicies and then check convergence. "
        # Find local minimas for middel part of line 
        i_minimas=np.where((fvalues[1:-1]<fvalues[:-2])&(fvalues[2:]>fvalues[1:-1]))[0]+1
        # Check if the function values for the minimums are within the tolerance
        if len(i_minimas):
            i_keep=abs(fvalues[i_minimas+1]+fvalues[i_minimas-1]-2.0*fvalues[i_minimas])>=self.ftol*(1.0+abs(fvalues[i_minimas]))
            i_minimas=i_minimas[i_keep]
        # Find local minimas for end parts of line
        if fvalues[0]-fvalues[1]<-self.ftol:
            i_minimas=np.append([1],i_minimas)
        if fvalues[-1]-fvalues[-2]<-self.ftol:
            i_minimas=np.append(i_minimas,[len_l-2])
        # Check the distances in the local minimas are within the tolerance
        if len(i_minimas):
            i_keep=abs(xvalues[i_minimas+1,theta_index]-xvalues[i_minimas-1,theta_index])>=self.xtol*(1.0+abs(xvalues[i_minimas,theta_index]))
            i_minimas=i_minimas[i_keep]
        # Sort the indicies after function value sizes
        if len(i_minimas)>1:
            i_sort=np.argsort(fvalues[i_minimas])
            i_minimas=i_minimas[i_sort]
        return i_minimas
    
    def find_single_min(self,xvalues,fvalues,i_min,len_l,theta_index,**kwargs):
        " Find the global minimum and then check convergence. "
        # Investigate the global minimum
        i_minimas=[]
        if i_min==0:
            # Check if the function values are converged if the endpoints is the minimum value
            if fvalues[0]-fvalues[1]<-self.ftol:
                i_minimas=[i_min+1]
        elif i_min==int(len_l-1):
            # Check if the function values are converged if the endpoints is the minimum value
            if fvalues[-1]-fvalues[-2]<-self.ftol:
                i_minimas=[i_min-1]
        else:
            # Check if the function values are converged
            if abs(fvalues[i_min+1]+fvalues[i_min-1]-2.0*fvalues[i_min])>=self.ftol*(1.0+abs(fvalues[i_min])):
                i_minimas=[i_min]
        # Check if the distance in the local minimum is converged
        if len(i_minimas):
            i_minima=i_minimas[0]
            if not abs(xvalues[i_minima+1,theta_index]-xvalues[i_minima-1,theta_index])>=self.xtol*(1.0+abs(xvalues[i_minima,theta_index])):
                i_minimas=[]
        return np.array(i_minimas)
    
    def get_theta_index(self,parameters=[],**kwargs):
        " Get the theta_index. "
        if self.theta_index is None:
            if 'length' in parameters:
                return list(parameters).index('length')
            return 0
        return self.theta_index

    def set_tols(self,tol,xtol=None,ftol=None,**kwargs):
        " Set xtol and ftol to the tolerance if they are not given. "
        if xtol is None:
            xtol=tol
        if ftol is None:
            ftol=tol
        return xtol,ftol

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,
                        tol=self.tol,
                        optimize=self.optimize,
                        multiple_min=self.multiple_min,
                        theta_index=self.theta_index,
                        parallel=self.parallel,
                        xtol=self.xtol,
                        ftol=self.ftol)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs

    
class GoldenSearch(LineSearchOptimizer):
    def __init__(self,maxiter=5000,tol=1e-5,optimize=True,multiple_min=True,theta_index=None,parallel=False,xtol=None,ftol=None,**kwargs):
        """
        The golden section search method is used as the line search optimizer.
        The line search optimizer is used for optimzing the objective function wrt. a single the hyperparameter.
        The GoldenSearch does only work together with a GlobalOptimizer that uses line searches (e.g. FactorizedOptimizer). 
        A line of the hyperparameter is required to run the line search.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        """
        super().__init__(maxiter=maxiter,
                         tol=tol,
                         optimize=optimize,
                         multiple_min=multiple_min,
                         theta_index=theta_index,
                         parallel=parallel,
                         xtol=xtol,
                         ftol=ftol,
                         **kwargs)

    def run(self,func,line,parameters,model,X,Y,pdis,**kwargs):
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,self.jac,**kwargs)
        # Get the index of the line optimized hyperparameter 
        theta_index=self.get_theta_index(parameters)
        # Calculate function values for line coordinates
        len_l=len(line)
        line=line.reshape(len_l,-1)
        f_list=self.calculate_values(line,func,func_args=func_args)
        # Find the optimal value
        i_min=np.nanargmin(f_list)
        sol={'fun':f_list[i_min],'x':line[i_min],'success':False,'nfev':len_l,'nit':len_l}
        # Check whether the object function is flat
        if (np.nanmax(f_list)-f_list[i_min])<self.ftol:
            i=int(np.floor(0.3*(len(line)-1)))
            return {'fun':f_list[i],'x':line[i],'success':False,'nfev':len_l,'nit':len_l}
        # Find local minimums or the global minimum
        i_minimas=self.find_minimas(line,f_list,i_min,len_l,theta_index)
        # Check for convergence 
        len_i=len(i_minimas)
        if len_i==0:
            sol['success']=True
            return sol
        # Do multiple golden section search if necessary
        if self.optimize and self.maxiter>2:
            sol=self.prepare_run_golden(func,sol,i_minimas,line,f_list,theta_index,func_args=func_args)
        return sol
    
    def prepare_run_golden(self,func,sol,i_minimas,line,f_list,theta_index,func_args=(),**kwargs):
        " Prepare and run the golden section search for the minimums. "
        niter=sol['nfev']
        # Get the function that evaluate the objective function
        fun=self.get_fun(func)
        for i_min in i_minimas:
            # Find the indicies of the interval
            x1=i_min-1
            x4=i_min+1
            # Get the function values of the endpoints of the interval
            f1,f4=f_list[x1],f_list[x4]
            # Get the initial vector as the lower interval coordinates
            theta0=line[x1].copy()
            # Get the direction vector to the upper interval coordinates
            direc=line[x4]-theta0
            # Calculate the norm of the direction vector for the important dimension
            direc_norm=abs(direc[theta_index])
            # Perform the golden section search in the interval
            sol_o=self.golden_search(fun,[0.0,1.0],fbracket=[f1,f4],
                                     maxiter=int(self.maxiter-niter),func_args=func_args,
                                     vec0=theta0,direc=direc,direc_norm=direc_norm)
            # Update the solution
            niter+=sol_o['nfev']
            if sol_o['fun']<=sol['fun']:
                sol=sol_o.copy()
            if niter>=self.maxiter:
                break
        sol['nfev'],sol['nit']=niter,niter
        return sol

    def golden_search(self,fun,bracket,maxiter=200,func_args=(),fbracket=None,vec0=np.array([0.0]),direc=np.array([1.0]),direc_norm=None,**kwargs):
        " Perform a golden section search. "
        # Golden ratio
        r=(np.sqrt(5)-1)/2
        c=1-r
        # Number of function evaluations
        nfev=0
        # Get the coordinates and function values of the endpoints of the interval
        x1,x4=bracket
        vec1=vec0+direc*x1
        vec4=vec0+direc*x4
        if fbracket is None:
            f1,f4=fun(vec1,*func_args),fun(vec4,*func_args)
            nfev+=2
        else:
            f1,f4=fbracket
        # Direction vector norm
        if direc_norm is None:
            direc_norm=np.linalg.norm(direc)
        # Check if the maximum number of iterations have been used
        if maxiter<3:
            i_min=np.nanargmin([f1,f4])
            sol={'fun':[f1,f4][i_min],'x':[vec1,vec4][i_min],'success':False,'nfev':nfev,'nit':nfev}
            return sol
        # Check if the convergence criteria is already met in terms of the coordinates
        if abs(x4-x1)*direc_norm<=self.xtol:
            i_min=np.nanargmin([f1,f4])
            sol={'fun':[f1,f4][i_min],'x':[vec1,vec4][i_min],'success':True,'nfev':nfev,'nit':nfev}
            return sol
        # Make and calculate points within the interval
        x_list=[x1,r*x1+c*x4,c*x1+r*x4,x4]
        f_list=[f1,fun(vec0+direc*x_list[1],*func_args),fun(vec0+direc*x_list[2],*func_args),f4]
        nfev+=2
        # Perform the line search
        success=False
        while nfev<maxiter:
            i_min=np.nanargmin(f_list)
            # Check for convergence
            if np.nanmax(f_list)-f_list[i_min]<=self.ftol*(1.0+abs(f_list[i_min])) or abs(x_list[3]-x_list[0])*direc_norm<=self.xtol*(1.0+direc_norm*abs(x_list[1])):
                success=True
                break
            # Calculate a new point 
            if i_min<2:
                x_list[3]=x_list[2]
                f_list[3]=f_list[2]
                x_list[2]=x_list[1]
                f_list[2]=f_list[1]
                x_list[1]=r*x_list[2]+c*x_list[0]
                f_list[1]=fun(vec0+direc*x_list[1],*func_args)
            else:
                x_list[0]=x_list[1]
                f_list[0]=f_list[1]
                x_list[1]=x_list[2]
                f_list[1]=f_list[2]
                x_list[2]=r*x_list[1]+c*x_list[3]
                f_list[2]=fun(vec0+direc*x_list[2],*func_args)
            nfev+=1
        # Get the solution
        i_min=np.nanargmin(f_list)
        sol={'fun':f_list[i_min],'x':vec0+direc*(x_list[i_min]),'success':success,'nfev':nfev,'nit':nfev}
        return sol


class FineGridSearch(LineSearchOptimizer):
    def __init__(self,maxiter=5000,tol=1e-5,optimize=True,multiple_min=True,ngrid=80,loops=3,theta_index=None,parallel=False,xtol=None,ftol=None,**kwargs):
        """
        The fine grid search method is used as the line search optimizer.
        The line search optimizer is used for optimzing the objective function wrt. a single the hyperparameter.
        Finer grids are made for all minimums of the objective function.
        The FineGridSearch does only work together with a GlobalOptimizer that uses line searches (e.g. FactorizedOptimizer).
        A line of the hyperparameter is required to run the line search.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            ngrid : int
                The number of grid points of the hyperparameter that is optimized.
            loops : int
                The number of loops where the grid points are made.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        """
        # Line search optimizers cannot use gradients of the objective function 
        self.jac=False
        # Set the default theta_index
        self.theta_index=None
        # Set xtol and ftol to the tolerance if they are not given.
        xtol,ftol=self.set_tols(tol,xtol=xtol,ftol=ftol)
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,
                              tol=tol,
                              optimize=optimize,
                              multiple_min=multiple_min,
                              ngrid=ngrid,
                              loops=loops,
                              theta_index=theta_index,
                              parallel=parallel,
                              xtol=xtol,
                              ftol=ftol,
                              **kwargs)

    def run(self,func,line,parameters,model,X,Y,pdis,**kwargs):
        # Get the function arguments
        func_args=self.get_func_arguments(parameters,model,X,Y,pdis,self.jac,**kwargs)
        # Get the index of the line optimized hyperparameter 
        theta_index=self.get_theta_index(parameters)
        # Make empty solution and lists
        sol=self.get_empty_solution()
        lines=np.empty((0,len(line[0])))
        f_lists=np.empty((0))
        # Get the solution from loops of the fine grid method
        sol=self.run_grid_loops(func,sol,line,
                                lines,f_lists,theta_index,
                                loops=self.loops,maxiter=self.maxiter,
                                func_args=func_args)
        return sol
    
    def update_arguments(self,maxiter=None,tol=None,optimize=None,multiple_min=None,ngrid=None,loops=None,theta_index=None,parallel=None,xtol=None,ftol=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            ngrid : int
                The number of grid points of the hyperparameter that is optimized.
            loops : int
                The number of loops where the grid points are made.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        Returns:
            self: The updated object itself.
        """
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if tol is not None:
            self.tol=tol
        if optimize is not None:
            self.optimize=optimize
        if multiple_min is not None:
            self.multiple_min=multiple_min
        if theta_index is not None:
            self.theta_index=int(theta_index)
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
        if loops is not None:
            self.loops=int(loops)
        if xtol is not None:
            self.xtol=xtol
        if ftol is not None:
            self.ftol=ftol
        return self

    def run_grid_loops(self,func,sol,line,lines,f_lists,theta_index,loops=3,maxiter=400,func_args=(),**kwargs):
        " Calculate finer grid points and find minimums in a loop if optimize=True. "
        # Calculate function values for line coordinates
        len_l=len(line)
        line=line.reshape(len_l,-1)
        f_list=self.calculate_values(line,func,func_args=func_args)
        # Use previously calculated grid points
        lines=np.append(lines,line,axis=0)
        i_sort=np.argsort(lines[:,theta_index])
        lines=lines[i_sort]
        f_lists=np.append(f_lists,f_list)[i_sort]
        # Find the minimum value
        i_min=np.nanargmin(f_lists)
        # Update the solution dictionary
        sol['nfev']+=len_l
        sol['nit']+=len_l
        if f_lists[i_min]<=sol['fun']:
            sol['fun']=f_lists[i_min]
            sol['x']=lines[i_min]
        # Find local minimums or the global minimum
        i_minimas=self.find_minimas(lines,f_lists,i_min,len(lines),theta_index)
        # Check for convergence
        len_i=len(i_minimas)
        if len_i==0:
            sol['success']=True
            return sol
        # Optimize the minimums
        if self.optimize and loops>0 and maxiter>self.ngrid:
            # Make a new grid if minimums exist
            newline,lines,f_lists=self.make_new_line(lines,f_lists,i_minimas,theta_index,len_i)
            return self.run_grid_loops(func,sol,newline,lines,f_lists,theta_index,loops=int(loops-1),maxiter=int(maxiter-len_l),func_args=func_args)
        return sol

    def make_new_line(self,lines,f_lists,i_minimas,theta_index,len_i,**kwargs):
        " Make a new line/grid for the minimums to optimize the hyperparameter. "
        # Find the grid points that must be saved for later
        i_all=(i_minimas+np.array([[-1],[0],[1]],dtype=int)).T.reshape(-1)
        saved_lines=lines[i_all]
        saved_f_lists=f_lists[i_all]
        # Make a new grid if minimums exist
        if self.multiple_min:
            # If 3 grid points can not be used per minumum then use the lowest minimums
            if self.ngrid<len_i*3:
                i_minimas=i_minimas[:self.ngrid//3]
                len_i=len(i_minimas)
            # Get the number of grid points for each minimum
            di=np.full(shape=len_i,fill_value=self.ngrid//len_i,dtype=int)
            # Get an extra grid point to the lowest minimums if there are grid points to spare
            di[:int(self.ngrid%len_i)]+=1
            # Make new line
            newline=np.concatenate([np.linspace(lines[i-1],lines[i+1],di[j]+2)[1:-1] for j,i in enumerate(i_minimas)])
        else:
            i_min=i_minimas[0]
            # Make new line
            newline=np.linspace(lines[i_min-1],lines[i_min+1],self.ngrid+2)[1:-1]
        return newline,saved_lines,saved_f_lists
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,
                        tol=self.tol,
                        optimize=self.optimize,
                        multiple_min=self.multiple_min,
                        ngrid=self.ngrid,
                        loops=self.loops,
                        theta_index=self.theta_index,
                        parallel=self.parallel,
                        xtol=self.xtol,
                        ftol=self.ftol)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
        

class TransGridSearch(FineGridSearch):
    def __init__(self,maxiter=5000,tol=1e-5,optimize=True,multiple_min=True,ngrid=80,loops=3,use_likelihood=True,theta_index=None,parallel=False,xtol=None,ftol=None,**kwargs):
        """
        The variable transformed grid search method is used as the line search optimizer.
        The line search optimizer is used for optimzing the objective function wrt. a single the hyperparameter.
        Grids are made by updating the variable transformation from the objective function values.
        The TransGridSearch does only work together with a GlobalOptimizer that uses line searches (e.g. FactorizedOptimizer).
        A line of the hyperparameter is required to run the line search.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            ngrid : int
                The number of grid points of the hyperparameter that is optimized.
            loops : int
                The number of loops where the grid points are made.
            use_likelihood : bool
                Whether to use the objective function as a log-likelihood or not.
                If the use_likelihood=False, the objective function is scaled and 
                shifted with the maximum value.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        """
        # Line search optimizers cannot use gradients of the objective function 
        self.jac=False
        # Set the default theta_index
        self.theta_index=None
        # Set xtol and ftol to the tolerance if they are not given.
        xtol,ftol=self.set_tols(tol,xtol=xtol,ftol=ftol)
        # Set all the arguments
        self.update_arguments(maxiter=maxiter,
                              tol=tol,
                              optimize=optimize,
                              multiple_min=multiple_min,
                              ngrid=ngrid,
                              loops=loops,
                              use_likelihood=use_likelihood,
                              theta_index=theta_index,
                              parallel=parallel,
                              xtol=xtol,
                              ftol=ftol,
                              **kwargs)
        
    def update_arguments(self,maxiter=None,tol=None,optimize=None,multiple_min=None,ngrid=None,loops=None,use_likelihood=None,theta_index=None,parallel=None,xtol=None,ftol=None,**kwargs):
        """
        Update the optimizer with its arguments. The existing arguments are used if they are not given.
        Parameters:
            maxiter : int
                The maximum number of evaluations or iterations the optimizer can use.
            tol : float
                A tolerance criterion for convergence. 
            optimize : bool
                Whether to optimize the line given by split it into smaller intervals. 
            multiple_min : bool
                Whether to optimize multiple minimums or just optimize the lowest minimum.
            ngrid : int
                The number of grid points of the hyperparameter that is optimized.
            loops : int
                The number of loops where the grid points are made.
            use_likelihood : bool
                Whether to use the objective function as a log-likelihood or not.
                If the use_likelihood=False, the objective function is scaled and 
                shifted with the maximum value.
            theta_index : int or None
                The index of the hyperparameter that is optimized with the line search.
                If theta_index=None, then it will use the index of the length-scale.
                If theta_index=None and no length-scale, then theta_index=0.
            parallel : bool
                Whether to calculate the grid points in parallel over multiple CPUs.
            xtol : float
                A tolerance criterion of the hyperparameter for convergence. 
            ftol : float
                A tolerance criterion of the objective function for convergence. 
        Returns:
            self: The updated object itself.
        """
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if tol is not None:
            self.tol=tol
        if optimize is not None:
            self.optimize=optimize
        if multiple_min is not None:
            self.multiple_min=multiple_min
        if theta_index is not None:
            self.theta_index=int(theta_index)
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
        if loops is not None:
            self.loops=int(loops)
        if use_likelihood is not None:
            self.use_likelihood=use_likelihood
        if xtol is not None:
            self.xtol=xtol
        if ftol is not None:
            self.ftol=ftol
        return self

    def make_new_line(self,lines,f_lists,i_minimas,theta_index,len_i,**kwargs):
        " Make new line/grid points from the variable transformation of the objective function. "
        from scipy.integrate import cumulative_trapezoid
        # Change the function to likelihood or to a scaled function from 0 to 1
        if self.use_likelihood:
            fs=np.exp(-(f_lists-np.nanmin(f_lists)))
        else:
            fs=-(f_lists-np.nanmax(f_lists))
            fs=fs/np.nanmax(fs)
        # Calculate the cumulative distribution function values on the grid
        cdf=cumulative_trapezoid(fs,x=lines[:,theta_index],initial=0.0)
        cdf=cdf/cdf[-1]
        cdf_r=cdf.reshape(-1,1)
        # Make new grid points on the inverse cumulative distribution function
        dl=np.finfo(float).eps
        newlines=np.linspace(0.0+dl,1.0-dl,self.ngrid)
        # Find the intervals where the new grid points are located
        i_new=np.where((cdf_r[:-1]<=newlines)&(newlines<cdf_r[1:]))[0]
        i_new_a=i_new+1
        # Calculate the linear interpolation for the intervals of interest
        slope=(lines[i_new_a]-lines[i_new])/(cdf_r[i_new_a]-cdf_r[i_new])
        intercept=((lines[i_new]*cdf_r[i_new_a])-(lines[i_new_a]*cdf_r[i_new]))/(cdf_r[i_new_a]-cdf_r[i_new])
        # Calculate the hyperparameters
        newline=slope*newlines.reshape(-1,1)+intercept
        return newline,lines,f_lists
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(maxiter=self.maxiter,
                        tol=self.tol,
                        optimize=self.optimize,
                        multiple_min=self.multiple_min,
                        ngrid=self.ngrid,
                        loops=self.loops,
                        use_likelihood=self.use_likelihood,
                        theta_index=self.theta_index,
                        parallel=self.parallel,
                        xtol=self.xtol,
                        ftol=self.ftol)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs

