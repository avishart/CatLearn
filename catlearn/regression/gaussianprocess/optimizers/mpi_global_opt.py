import numpy as np
from scipy.optimize import OptimizeResult
from .local_opt import scipy_opt,fine_grid_search
from .functions import make_lines,make_grid,sample_thetas

def calculate_list_values_parallelize(line,fun,*args,comm=None,**kwargs):
    " Calculate a list of values with a function in parallel. "
    # Setup parallel run 
    rank,size=comm.rank,comm.size
    f_list=np.array([fun(theta,*args) for t,theta in enumerate(line) if rank==t%size])
    return np.array([comm.bcast(f_list,root=r) for r in range(size)]).T.reshape(-1)

def get_solution_parallelized(sol,fun,parameters,model,X,Y,pdis,comm,size,**kwargs):
    " Get all solutions from each function at each rank. "
    sol=fun.get_solution(sol,parameters,model,X,Y,pdis)
    sols=[comm.bcast(sol,root=r) for r in range(size)]
    i_min=np.argmin([sols[r]['fun'] for r in range(size)])
    return sols[i_min]

def local_optimize_parallel(fun,thetas,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,local_kwargs={},comm=None,**global_kwargs):
    " Local optimization of the hyperparameters in parallel. "
    args=(parameters,model,X,Y,pdis,jac)
    rank,size=comm.rank,comm.size
    nfev=0
    sol={'fun':np.inf,'x':np.array([]),'success':False,'nfev':0,'nit':0,'message':"No function values calculated."}
    sols_p=[local_run(fun.function,theta,jac=jac,maxiter=maxiter,args=args,**local_kwargs) for t,theta in enumerate(thetas) if rank==t%size]
    for r in range(size):
        sols=comm.bcast(sols_p,root=r)
        for solp in sols:
            if solp['fun']<sol['fun']:
                sol=solp.copy()
            nfev+=solp['nfev']
    sol['nfev'],sol['nit']=nfev,nfev
    return OptimizeResult(**sol)

def random_parallel(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,npoints=50,hptrans=True,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    " Sample and optimize npoints random points in the variable transformation region in parallel. "
    # Setup parallel run 
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    size=comm.size
    # Make sure the number of points is paralizable
    npoints=int(int(npoints/size)*size)
    if npoints==0:
        npoints=size
    # Draw random hyperparameter samples
    if npoints>1:
        thetas=sample_thetas(parameters,model,X,Y,bounds=bounds,npoints=npoints,hptrans=hptrans,use_bounds=use_bounds,s=s)
        thetas=np.append(np.array([x0]),thetas,axis=0)
    else:
        thetas=np.array([x0])
    # Perform the local optimization for random samples in parallel
    sol=local_optimize_parallel(fun,thetas,parameters,model,X,Y,pdis,local_run=local_run,maxiter=maxiter,jac=jac,local_kwargs=local_kwargs,comm=comm,**global_kwargs)
    # Get all solutions from each rank
    sol=get_solution_parallelized(sol,fun,parameters,model,X,Y,pdis,comm,size)
    return OptimizeResult(**sol)

def grid_parallel(fun,x0,parameters,model,X,Y,pdis,local_run=scipy_opt,maxiter=5000,jac=True,bounds=None,n_each_dim=None,hptrans=True,optimize=False,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    "Make a brute-force grid optimization of the hyperparameters in parallel. "
    # Setup parallel run 
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    size=comm.size
    # Number of points per dimension
    dim=len(x0)
    maxiter=int(int(maxiter/size)*size)
    if n_each_dim is None or np.prod(n_each_dim)%size!=0:
        n_each_dim=int(int(maxiter-1)**(1/dim))
        n_each_dim=n_each_dim if n_each_dim>1 else 1
    # Make grid either with the same or different numbers in each dimension
    lines=make_lines(parameters,model,X,Y,bounds=bounds,ngrid=n_each_dim,hptrans=hptrans,use_bounds=use_bounds,ngrid_each_dim=isinstance(n_each_dim,(list,np.ndarray)),s=s)
    theta_r=np.array(make_grid(lines,int(maxiter-1)))
    theta_r=np.append(np.array([x0]),theta_r,axis=0)
    # Calculate the grid points
    args=(parameters,model,X,Y,pdis,False)
    f_list=calculate_list_values_parallelize(theta_r,fun.function,*args,comm=comm)
    i_min=np.nanargmin(f_list)
    sol={'fun':f_list[i_min],'x':theta_r[i_min],'success':False,'nfev':len(f_list),'nit':len(f_list),'message':"Grid points are calculated."}
    # Get all solutions from each rank
    sol=get_solution_parallelized(sol,fun,parameters,model,X,Y,pdis,comm,size)
    return OptimizeResult(**sol)

def line_search_scale_parallel(fun,x0,parameters,model,X,Y,pdis,local_run=fine_grid_search,maxiter=5000,jac=False,bounds=None,ngrid=80,hptrans=True,use_bounds=True,s=0.14,local_kwargs={},**global_kwargs):
    " Do a scaled line search in 1D (made for the 1D length scale) in parallel. "
    # Setup parallel run 
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    size=comm.size
    ## Check that all important arguments are set right for parallel run
    if 'fun_list' not in local_kwargs.keys():
        local_kwargs['fun_list']=calculate_list_values_parallelize
    if 'fun_kwargs' not in local_kwargs.keys():
        local_kwargs['fun_kwargs']={}
    local_kwargs['fun_kwargs']['comm']=comm
    ### Make sure the number of grid points is parallelizable and not zero
    ngrid=int(int(ngrid/size)*size)
    if ngrid==0:
        ngrid=int(size)
    if 'iterloop' not in local_kwargs.keys():
        local_kwargs['iterloop']=80
    local_kwargs['iterloop']=int(int(local_kwargs['iterloop']/size)*size)
    if local_kwargs['iterloop']==0:
        local_kwargs['iterloop']=int(size)
    # Find the index of the length-scale
    if 'theta_index' not in local_kwargs.keys() and 'length' in parameters:
        local_kwargs['theta_index']=list(parameters).index('length')
    # Stationary arguments
    args=(parameters,model,X,Y,pdis,False)
    sol={'fun':np.inf,'x':np.array([]),'success':False,'nfev':0,'nit':0,'message':"No function values calculated."}
    # Calculate all points on line
    lines=np.array(make_lines(parameters,model,X,Y,bounds=bounds,ngrid=ngrid,hptrans=hptrans,use_bounds=use_bounds,ngrid_each_dim=False,s=s)).T
    sol=local_run(fun.function,lines,maxiter=maxiter,args=args,**local_kwargs)
    # Get all solutions from each rank
    sol=get_solution_parallelized(sol,fun,parameters,model,X,Y,pdis,comm,size)
    if sol['success']:
        sol['message']='Local optimization is converged.'
    else:
        sol['message']='Local optimization is not converged.'
    return OptimizeResult(**sol)

