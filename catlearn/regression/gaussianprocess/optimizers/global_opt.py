import numpy as np
from scipy.optimize import OptimizeResult,basinhopping,dual_annealing
from .local_opt import scipy_opt,run_golden,fine_grid_search
from .functions import hp_to_theta,make_grid,make_lines,sample_thetas,anneal_var_trans

def function(fun,x0,parameters,model,X,Y,pdis,jac=True,**global_kwargs):
    " Function value of single point calculation of the objective function.  "
    f=fun.function(x0,parameters,model,X,Y,pdis,jac)
    sol={'x':x0,'success':False,'nfev':1,'nit':1,'message':"Function value is calculated."}
    if jac:
        sol['fun'],sol['jac']=f
    else:
        sol['fun']=f
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return OptimizeResult(**sol)

def local_optimize(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},**global_kwargs):
    " Local optimization of the hyperparameters. "
    args=(parameters,model,X,Y,pdis,jac)
    sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return OptimizeResult(**sol)

def local_prior(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},**global_kwargs):
    " Local optimization "
    args=(parameters,model,X,Y,pdis,jac)
    sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    if pdis is not None:
        niter=sol['nfev']
        args=(parameters,model,X,Y,None,jac)
        sol=local_run(fun.function,sol['x'],jac=jac,maxiter=maxiter,args=args,**local_kwargs)
        sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
        sol['nfev']+=niter
    sol['nit']=2
    return OptimizeResult(**sol)

def local_ed_guess(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},**global_kwargs):
    " Local optimization for initial and educated guess hyperparameter sets "
    args=(parameters,model,X,Y,pdis,jac)
    sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    from ..educated import Educated_guess
    hp_ed=Educated_guess(prior=model.prior,kernel=model.kernel,parameters=parameters).get_hp(X,Y,parameters=parameters)
    x_ed,parameters=hp_to_theta(hp_ed)
    sol_ed=local_run(fun.function,x_ed,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    if sol['fun']<sol_ed['fun']:
        sol['nfev']+=sol_ed['nfev']
        sol['nit']=2
        sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
        return OptimizeResult(**sol)
    sol_ed['nfev']+=sol['nfev']
    sol_ed['nit']=2
    sol_ed=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return OptimizeResult(**sol_ed)

def random(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,npoints=50,hptrans=True,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    " Sample and optimize npoints random points in the variable transformation region "
    args=(parameters,model,X,Y,pdis,jac)
    # Draw random hyperparameter samples
    if npoints>1:
        thetas=sample_thetas(parameters,model,X,Y,bounds=bounds,npoints=int(npoints-1),hptrans=hptrans,use_bounds=use_bounds,s=s)
        thetas=np.append(np.array([x0]),thetas,axis=0)
    else:
        thetas=np.array([x0])
    # Perform the local optimization for random samples
    sol={'fun':np.inf,'x':np.array([]),'success':False,'nfev':0,'nit':0,'message':"No function values calculated."}
    nfev=0
    for theta in thetas:
        if nfev>=maxiter:
            break
        sol_s=local_run(fun.function,theta,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
        if sol_s['fun']<sol['fun']:
            sol.update(sol_s)
        nfev+=sol_s['nfev']
    sol['nfev'],sol['nit']=nfev,len(thetas)
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return OptimizeResult(**sol)

def grid(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,n_each_dim=None,hptrans=True,optimize=True,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    "Make a brute-force grid optimization of the hyperparameters"
    # Number of points per dimension
    dim=len(x0)
    if n_each_dim is None:
        n_each_dim=int(maxiter**(1/dim))
        n_each_dim=n_each_dim if n_each_dim>1 else 1
    # Make grid either with the same or different numbers in each dimension
    lines=make_lines(parameters,model,X,Y,bounds=bounds,ngrid=n_each_dim,hptrans=hptrans,use_bounds=use_bounds,ngrid_each_dim=isinstance(n_each_dim,(list,np.ndarray)),s=s)
    theta_r=np.array(make_grid(lines,maxiter-1))
    # Set the calculator up 
    args=(parameters,model,X,Y,pdis,False)
    sol={'fun':fun.function(x0,*args),'x':x0,'success':False,'nfev':1,'nit':1,'message':"Initial hyperparameters used."}
    nfev=1
    # Calculate the grid points
    for theta in theta_r:
        f=fun.function(theta,*args)
        if f<sol['fun']:
            sol['fun'],sol['x']=f,theta
            sol['message']="Lower function value found."
    nfev+=len(theta_r)
    # Local optimize the best point if wanted
    local_maxiter=int(maxiter-nfev)
    local_maxiter=0 if local_maxiter<0 else local_maxiter
    if optimize:
        args=(parameters,model,X,Y,pdis,jac)
        mopt=local_run(fun.function,sol['x'],jac=jac,maxiter=int(maxiter-nfev),args=args,**local_kwargs)
        if mopt['fun']<=sol['fun']:
            sol=mopt.copy()
        nfev+=mopt['nfev']
    sol['nfev'],sol['nit']=nfev,nfev
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return OptimizeResult(**sol)

def line(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,n_each_dim=None,hptrans=True,loops=3,optimize=True,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    "Make a linesearch in each of the dimensions of the hyperparameters iteratively"
    # Number of points per dimension
    dim=len(x0)
    if n_each_dim is None or np.sum(n_each_dim)*loops>maxiter:
        n_each_dim=int(maxiter/(loops*dim))
        n_each_dim=n_each_dim if n_each_dim>1 else 1
    # Make grid either with the same or different numbers in each dimension
    lines=make_lines(parameters,model,X,Y,bounds=bounds,ngrid=n_each_dim,hptrans=hptrans,use_bounds=use_bounds,ngrid_each_dim=isinstance(n_each_dim,(list,np.ndarray)),s=s)
    # Set the calculator up 
    args=(parameters,model,X,Y,pdis,False)
    sol={'fun':fun.function(x0,*args),'x':x0,'success':False,'nfev':1,'nit':1,'message':"Initial hyperparameters used."}
    nfev=1
    # Calculate the line points
    for l in range(int(loops)):
        dim_perm=np.random.permutation(list(range(dim)))
        for d in dim_perm:
            for theta_d in lines[d]:
                theta_r=sol['x'].copy()
                theta_r[d]=theta_d
                f=fun.function(theta_r,*args)
                if f<sol['fun']:
                    sol['fun'],sol['x']=f,theta_r.copy()
                    sol['message']="Lower function value found."
                nfev+=1
    # Local optimize the best point if wanted
    if optimize:
        args=(parameters,model,X,Y,pdis,jac)
        mopt=local_run(fun.function,sol['x'],jac=jac,maxiter=int(maxiter-nfev),args=args,**local_kwargs)
        if mopt['fun']<=sol['fun']:
            sol=mopt.copy()
        nfev+=mopt['nfev']
    sol['nfev'],sol['nit']=nfev,nfev
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return OptimizeResult(**sol)

def basin(fun,x0,parameters,model,X,Y,pdis,maxiter=5000,jac=True,niter=5,interval=10,T=1.0,stepsize=0.1,niter_success=None,local_kwargs={},**global_kwargs):
    " Basin-hopping optimization of the hyperparameters "
    # Set the local optimizer parameter
    if 'options' in local_kwargs.keys():
        local_kwargs['options']['maxiter']=int(maxiter/niter)
    else:
        local_kwargs['options']=dict(maxiter=int(maxiter/niter))
    args=(parameters,model,X,Y,pdis,jac)
    minimizer_kwargs=dict(args=args,jac=jac,**local_kwargs)
    # Do the basin-hopping
    sol=basinhopping(fun.function,x0=x0,niter=niter,interval=interval,T=T,stepsize=stepsize,niter_success=niter_success,minimizer_kwargs=minimizer_kwargs)
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return sol

def annealling(fun,x0,parameters,model,X,Y,pdis,maxiter=5000,jac=False,bounds=None,initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False,use_bounds=True,local_kwargs={},**global_kwargs):
    " Dual simulated annealing optimization of the hyperparameters "
    # Arguments for this method
    options_dual=dict(initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False)
    local_kwargs['jac']=jac
    # Do the simulated annealing
    if bounds is None:
        from ..hptrans import Variable_Transformation
        from .functions import theta_to_hp
        hyper_var=Variable_Transformation().transf_para(parameters,model,X,Y,use_bounds=use_bounds)
        bounds=np.array([[1e-9,1.00-1e-9]]*len(x0))
        args=(fun,hyper_var,parameters,model,X,Y,pdis,jac)
        sol=dual_annealing(anneal_var_trans,bounds=bounds,args=args,maxiter=maxiter,maxfun=maxiter,minimizer_kwargs=local_kwargs,**options_dual)        
        sol['x']=hp_to_theta(hyper_var.transform_t_to_theta(theta_to_hp(sol['x'],parameters)))[0]
    else:
        args=(parameters,model,X,Y,pdis,jac)
        sol=dual_annealing(fun.function,bounds=bounds,args=args,maxiter=maxiter,maxfun=maxiter,minimizer_kwargs=local_kwargs,**options_dual)
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    return sol

def line_search_scale(fun,x0,parameters,model,X,Y,pdis,local_run=run_golden,maxiter=5000,jac=False,bounds=None,ngrid=80,hptrans=True,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    " Do a scaled line search in 1D (made for the 1D length-scale) "
    # Stationary arguments
    args=(parameters,model,X,Y,pdis,False)
    sol={'fun':fun.function(x0,*args),'x':x0,'success':False,'nfev':1,'nit':1,'message':"Initial hyperparameters used."}
    # Find the index of the length-scale
    if 'theta_index' not in local_kwargs.keys() and 'length' in parameters:
        local_kwargs['theta_index']=list(parameters).index('length')
    # Calculate all points on line
    lines=np.array(make_lines(parameters,model,X,Y,bounds=bounds,ngrid=ngrid,hptrans=hptrans,use_bounds=use_bounds,ngrid_each_dim=False,s=s)).T
    sol=local_run(fun.function,lines,maxiter=maxiter,args=args,**local_kwargs)
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    if sol['success']:
        sol['message']='Local optimization is converged.'
    else:
        sol['message']='Local optimization is not converged.'
    return OptimizeResult(**sol)
