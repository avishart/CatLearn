import numpy as np
import copy

class HyperparameterFitter:

    def __init__(self,func=None,opt_algo=None,bounds=None,distance_matrix=True):
        """ Object with local and global optimization methods for optimizing the hyperparameters on different object functions 
        Parameters:
            func : class
                A class with the object function used to optimize the hyperparameters.
            opt_algo : class
                A class with the optimization method used.
            bounds : class or (D,2) numpy.array
                The boundary range for the optimization.
            distance_matrix : bool
                Whether to reuse the distance matrix for the optimization.
        """
        if func is None:
            from catlearn.regression.gp_bv.object_functions import Object_functions
            func=Object_functions(fun='nmll',log=True)
        if opt_algo is None:
            from catlearn.regression.gp_bv.opt_algorithm import Optimization_algorithm
            opt_kwargs={'npoints':100,'nopt':50,'get_ed_guess':True,'dis_min':True,'stop_criteria':10}
            opt_algo=Optimization_algorithm(opt_method='random',**opt_kwargs)
        if bounds is None:
            from catlearn.regression.gp_bv.bounds import Boundary_conditions
            bounds=Boundary_conditions(bound_type='restricted',scale=1)
        self.func=func
        self.opt_algo=opt_algo
        self.bounds=bounds
        self.distance_matrix=distance_matrix

    def fit(self,X,Y,GP,hp=None,maxiter=None,prior=None):
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
            maxiter : int
                Maximum number of iterations used by local or global optimization method.
            prior : dict
                A dict of prior distributions for each hyperparameter
        """
        # Linear or logarithmic space for the hyperparameters
        log=self.func.log
        # Make the hyperparameters ready for optimization 
        parameters,parameters_set,theta=self.make_parameters(hp,log,GP,len(X[0]))
        # Default maximum iterations, which is dependent on the dimension of the hyperparameters
        if maxiter is None:
            maxiter=1000*len(theta)
        # Update prior mean for GP
        GP.prior.update(X,Y[:,0])
        # Make the boundaries 
        if isinstance(self.bounds,np.ndarray):
            bounds=self.bounds.copy()
        else:
            bounds=self.bounds.create(GP,X,Y,parameters,log,self.func.fun_name)
        # Whether to use distance matrix
        if self.distance_matrix:
            dis_m=GP.kernel.dist_m(X)
        else:
            dis_m=None
        # Parameters for the optimization algorithm
        kwargs=copy.deepcopy(self.opt_algo.kwargs)
        self.opt_algo.fun_name=self.func.fun_name
        # Run the optimization
        sol=self.opt_algo.run(self.func.fun,GP,X,Y,theta,parameters,bounds=bounds,maxiter=int(maxiter),log=log,prior=prior,dis_m=dis_m,**kwargs)
        # If nmlp is chosen with nmll afterwards, then local optimization is performed in the end
        if 'nmlp+nmll'==self.func.fun_name:
            sol=self.opt_without_prior(sol,GP,X,Y,parameters,maxiter,log,prior,dis_m,kwargs)
        elif 'mnll'==self.func.fun_name or 'mnlp'==self.func.fun_name:
            return self.get_alphamax(sol,GP,parameters,X,Y,log=log,dis_m=dis_m)
        # Get the solution in right form
        sol=self.end_sol(sol,log,parameters,parameters_set)
        return sol

    def make_parameters(self,hp,log,GP,dim):
        " Make the hyperparameter values and names ready for optimization "
        # Get initial hyperparameter values if not given
        if hp is None:
            hp=copy.deepcopy(GP.hp)
        else:
            hp=copy.deepcopy(hp)
        # Make sure length hyperparameter in hp has the rigtht length for multiple length scales
        if 'length' in hp:
            if 'Multi' in str(GP.kernel):
                hp['length']=np.array(hp['length']).reshape(-1)
                if len(hp['length'])!=dim:
                    hp['length']=np.array([hp['length'][0]]*dim)
        # Sort the parameters the same way everytime
        parameters_set=sorted(list(set(hp.keys())))
        if 'correction' in parameters_set:
            parameters_set.remove('correction')
        theta=[list(np.array(hp[para]).reshape(-1)) for para in parameters_set]
        parameters=sum([[para]*len(theta[p]) for p,para in enumerate(parameters_set)],[])
        theta=np.array(sum(theta,[]))
        # If log scale is used (log hp must not be given!)
        theta=np.abs(theta)
        if log:
            theta=np.log(theta)
        return parameters,parameters_set,theta

    def end_sol(self,sol,log,parameters,parameters_set):
        "Make the solution from the local or global optimization into the right form"
        # Get the final hyperparameters in linear space
        if log:
            sol['x']=np.exp(sol['x'])
        else:
            sol['x']=np.abs(sol['x'])

        # Get the final hyperparameters
        hp={para:sol['x'][np.where(np.array(parameters)==para)[0]] for para in parameters_set}
        sol['hp']=hp
        return sol
    
    def get_alphamax(self,sol,GP,parameters,X,Y,log=False,dis_m=None):
        " Calculate alpha from maximizing LML and then the respective noise from optimization "
        # Get the final hyperparameters in linear space
        if log:
            sol['x']=np.exp(sol['x'])
        else:
            sol['x']=np.abs(sol['x'])
        GP=copy.deepcopy(GP)
        # Calculate maximized alpha value
        GP,parameters_set,sign_t=self.func.update_gp(GP,parameters,sol['x'])
        GP.set_hyperparams({'alpha':np.array([1.0])})
        coef,L,low,Y,KXX,n_data=self.func.get_coef(GP,X,Y,dis_m)
        alphamax=np.sqrt(np.matmul(Y.T,coef).item(0)/n_data)
        # Update alpha and transform the relative noise to the real noise
        sol['hp']={para:sol['x'][np.where(np.array(parameters)==para)[0]] for para in parameters_set}
        if 'alpha' in parameters:
            sol['x']=np.where(np.array(parameters)=='alpha',alphamax,sol['x'])
            sol['hp']['alpha']=np.array([alphamax]*len(sol['hp']['alpha']))
        else:
            sol['hp']['alpha']=np.array([alphamax])
        if 'noise' in parameters:
            sol['x']=np.where(np.array(parameters)=='noise',alphamax*sol['x'],sol['x'])
            sol['hp']['noise']=alphamax*sol['hp']['noise']
        else:
            sol['hp']['noise']=alphamax*GP.hp['noise']
        sol['nfev']+=1
        return sol

    def opt_without_prior(self,sol,GP,X,Y,parameters,maxiter,log,prior,dis_m,kwargs):
        " Do a local optimization without prior distributions "
        self.func.fun_choice('nmll',log)
        maxiter=int(maxiter-sol['nfev'])
        maxiter=100 if maxiter<100 else maxiter
        sol_l=self.opt_algo.local(self.func.fun,GP,X,Y,sol['x'],parameters,bounds=None,maxiter=maxiter,log=log,prior=prior,dis_m=dis_m,**kwargs)
        sol_l['nfev']+=sol['nfev']
        sol.update(sol_l)
        return sol








    










