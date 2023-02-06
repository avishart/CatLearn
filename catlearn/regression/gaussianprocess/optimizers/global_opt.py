import numpy as np
from scipy.optimize import OptimizeResult,basinhopping,dual_annealing
from .local_opt import scipy_opt,run_golden
from .functions import hp_to_theta,make_grid,anneal_var_trans

def function(fun,x0,gp,parameters,X,Y,prior,dis_m,jac=True,**global_kwargs):
    " Function value of single point "
    args=(gp,parameters,X,Y,prior,jac,dis_m)
    sol={'success':False,'x':x0,'nfev':1,'maxiter':1,'fun':fun.function(x0,*args)}
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

def local(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},**global_kwargs):
    " Local optimization "
    args=(gp,parameters,X,Y,prior,jac,dis_m)
    sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

def local_prior(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},**global_kwargs):
    " Local optimization "
    args=(gp,parameters,X,Y,prior,jac,dis_m)
    sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    sol=fun.get_solution(sol,*args)
    if prior is not None:
        niter=sol['nfev']
        args=(gp,parameters,X,Y,None,jac,dis_m)
        sol=local_run(fun.function,sol['x'],jac=jac,maxiter=maxiter,args=args,**local_kwargs)
        sol=fun.get_solution(sol,*args)
        sol['nfev']+=niter
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

def local_ed_guess(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},**global_kwargs):
    " Local optimization for initial and educated guess hyperparameter sets "
    args=(gp,parameters,X,Y,prior,jac,dis_m)
    sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    from ..educated import Educated_guess
    hp_ed=Educated_guess(gp).hp(X,Y,parameters=parameters)
    x_ed,parameters=hp_to_theta(hp_ed)
    sol_ed=local_run(fun.function,x_ed,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
    if sol['fun']<sol_ed['fun']:
        sol['nfev']+=sol_ed['nfev']
        sol=fun.get_solution(sol,*args)
        return OptimizeResult(**sol)
    sol_ed['nfev']+=sol['nfev']
    sol_ed=fun.get_solution(sol_ed,*args)
    return OptimizeResult(**sol_ed)

def random(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,npoints=50,use_bounds=True,local_kwargs={},**global_kwargs):
    " Sample and optimize npoints random points in the variable transformation region "
    args=(gp,parameters,X,Y,prior,jac,dis_m)
    if bounds is None:
        from ..hptrans import Variable_Transformation
        hyper_var=Variable_Transformation().transf_para(parameters,gp,X,Y,use_bounds=use_bounds)
    for n in range(npoints):
        if n==0:
            sol=local_run(fun.function,x0,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
            nfev=sol['nfev']
            continue
        if nfev>=maxiter:
            break
        if bounds is None:
            t={para:np.array([np.random.uniform(0.0,1.0)]) for para in parameters}
            hp_t=hyper_var.transform_t_to_theta(t)
            theta=hp_to_theta(hp_t)[0]
        else:
            theta=np.random.uniform(low=bounds[:,0],high=bounds[:,1],size=len(x0))
        sol_s=local_run(fun.function,theta,jac=jac,maxiter=maxiter,args=args,**local_kwargs)
        if sol_s['fun']<sol['fun']:
            sol.update(sol_s)
        nfev+=sol_s['nfev']
    sol['nfev']=nfev
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

def grid(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,n_each_dim=None,optimize=True,use_bounds=True,local_kwargs={},**global_kwargs):
    "Make a brute-force grid optimization of the hyperparameters"
    # Number of points per dimension
    dim=len(x0)
    if n_each_dim is None:
        n_each_dim=int(maxiter**(1/dim))
        n_each_dim=n_each_dim if n_each_dim>1 else 1
    if isinstance(n_each_dim,int):    
        n_each_dim=[n_each_dim]*dim
    # Make grid either with the same or different numbers in each dimension
    if bounds is None:
        from ..hptrans import Variable_Transformation
        hyper_var=Variable_Transformation().transf_para(parameters,gp,X,Y,use_bounds=use_bounds)
        dl=np.finfo(float).eps
        lines=[np.linspace(0.0+dl,1.0-dl,n_each_dim[p]) for p in range(dim)]
        lines=hyper_var.t_to_theta_lines(lines,parameters)
    else:
        lines=[np.linspace(bounds[p][0],bounds[p][1],n_each_dim[p]) for p in range(dim)]
    theta_r=np.array(make_grid(lines,maxiter-1))
    # Set the calculator up 
    args=(gp,parameters,X,Y,prior,False,dis_m)
    sol={'fun':fun.function(x0,*args),'x':x0,'success':False}
    nfev=1
    # Calculate the grid points
    for t in theta_r:
        f=fun.function(t,*args)
        if f<sol['fun']:
            sol['fun'],sol['x'],sol['success']=f,t,True
    nfev+=len(theta_r)
    # Local optimize the best point if wanted
    local_maxiter=int(maxiter-nfev)
    local_maxiter=0 if local_maxiter<0 else local_maxiter
    if optimize:
        args=(gp,parameters,X,Y,prior,jac,dis_m)
        mopt=local_run(fun.function,sol['x'],jac=jac,maxiter=int(maxiter-nfev),args=args,**local_kwargs)
        if mopt['fun']<=sol['fun']:
            sol=mopt.copy()
        nfev+=mopt['nfev']
    sol['nfev']=nfev
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

def line(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,n_each_dim=None,loops=3,optimize=True,use_bounds=True,local_kwargs={},**global_kwargs):
    "Make a linesearch in each of the dimensions of the hyperparameters iteratively"
    # Number of points per dimension
    dim=len(x0)
    if n_each_dim is None or np.sum(n_each_dim)*loops>maxiter:
        n_each_dim=int(maxiter/(loops*dim))
        n_each_dim=n_each_dim if n_each_dim>1 else 1
    if isinstance(n_each_dim,int):    
        n_each_dim=[n_each_dim]*dim
    # Make grid either with the same or different numbers in each dimension
    if bounds is None:
        from ..hptrans import Variable_Transformation
        hyper_var=Variable_Transformation().transf_para(parameters,gp,X,Y,use_bounds=use_bounds)
        dl=np.finfo(float).eps
        lines=[np.linspace(0.0+dl,1.0-dl,n_each_dim[p]) for p in range(dim)]
        lines=hyper_var.t_to_theta_lines(lines,parameters)
    else:
        lines=[np.linspace(bounds[p][0],bounds[p][1],n_each_dim[p]) for p in range(dim)]
    # Set the calculator up 
    args=(gp,parameters,X,Y,prior,False,dis_m)
    sol={'fun':fun.function(x0,*args),'x':x0,'success':False}
    nfev=1
    # Calculate the line points
    for l in range(int(loops)):
        dim_perm=np.random.permutation(list(range(dim)))
        for d in dim_perm:
            for t in lines[d]:
                theta_r=sol['x'].copy()
                theta_r[d]=t
                f=fun.function(theta_r,*args)
                if f<sol['fun']:
                    sol['fun'],sol['x'],sol['success']=f,theta_r.copy(),True
                nfev+=1
    # Local optimize the best point if wanted
    if optimize:
        args=(gp,parameters,X,Y,prior,jac,dis_m)
        mopt=local_run(fun.function,sol['x'],jac=jac,maxiter=int(maxiter-nfev),args=args,**local_kwargs)
        if mopt['fun']<=sol['fun']:
            sol=mopt
        nfev+=mopt['nfev']
    sol['nfev']=nfev
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

def basin(fun,x0,gp,parameters,X,Y,prior,dis_m,maxiter=5000,jac=True,niter=5,interval=10,T=1.0,stepsize=0.1,niter_success=None,local_kwargs={},**global_kwargs):
    " Basin-hopping optimization of the hyperparameters "
    # Set the local optimizer parameter
    if 'options' in local_kwargs.keys():
        local_kwargs['options']['maxiter']=int(maxiter/niter)
    else:
        local_kwargs['options']=dict(maxiter=int(maxiter/niter))
    args=(gp,parameters,X,Y,prior,jac,dis_m)
    minimizer_kwargs=dict(args=args,jac=jac,**local_kwargs)
    # Do the basin-hopping
    sol=basinhopping(fun.function,x0=x0,niter=niter,interval=interval,T=T,stepsize=stepsize,niter_success=niter_success,minimizer_kwargs=minimizer_kwargs)
    sol=fun.get_solution(sol,*args)
    return sol

def annealling(fun,x0,gp,parameters,X,Y,prior,dis_m,maxiter=5000,jac=False,bounds=None,initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False,use_bounds=True,local_kwargs={},**global_kwargs):
    " Dual simulated annealing optimization of the hyperparameters "
    # Arguments for this method
    options_dual=dict(initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False)
    local_kwargs['jac']=jac
    # Do the simulated annealing
    if bounds is None:
        from ..hptrans import Variable_Transformation
        from .functions import theta_to_hp
        hyper_var=Variable_Transformation().transf_para(parameters,gp,X,Y,use_bounds=use_bounds)
        bounds=np.array([[0.00,1.00]]*len(x0))
        args=(fun,hyper_var,gp,parameters,X,Y,prior,False,dis_m)
        sol=dual_annealing(anneal_var_trans,bounds=bounds,args=args,maxiter=maxiter,maxfun=maxiter,local_search_options=local_kwargs,**options_dual)        
        sol['x']=hp_to_theta(hyper_var.transform_t_to_theta(theta_to_hp(sol['x'],parameters)))[0]
    else:
        args=(gp,parameters,X,Y,prior,False,dis_m)
        sol=dual_annealing(fun.function,bounds=bounds,args=args,maxiter=maxiter,maxfun=maxiter,local_search_options=local_kwargs,**options_dual)
    sol=fun.get_solution(sol,gp,parameters,X,Y,prior,False,dis_m)
    return sol

def line_search_scale(fun,x0,gp,parameters,X,Y,prior,dis_m,local_run=run_golden,maxiter=5000,jac=False,bounds=None,ngrid=80,use_bounds=True,local_kwargs={},**global_kwargs):
    " Do a scaled line search in 1D (made for the 1D length scale) "
    # Stationary arguments
    args=(gp,parameters,X,Y,prior,False,dis_m)
    sol={'success':False,'x':x0,'nfev':1,'fun':fun.function(x0,*args)}
    # Calculate all points on line
    if bounds is None:
        from ..hptrans import Variable_Transformation
        hyper_var=Variable_Transformation().transf_para(parameters,gp,X,Y,use_bounds=use_bounds)
        dl=np.finfo(float).eps
        lines=[np.linspace(0.0+dl,1.0-dl,ngrid) for p in range(len(x0))]
    else:
        lines=[np.linspace(bounds[p][0],bounds[p][1],ngrid) for p in range(len(x0))]
    lines=hyper_var.t_to_theta_lines(lines,parameters).T
    sol=local_run(fun.function,lines,maxiter=maxiter,args=args,**local_kwargs)
    sol=fun.get_solution(sol,*args)
    return OptimizeResult(**sol)

