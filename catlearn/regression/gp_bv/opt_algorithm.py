import numpy as np
import copy
from numpy.lib.function_base import _DIMENSION_NAME
from scipy.linalg import cho_factor,cho_solve,cholesky,inv,solve_triangular
from scipy.optimize import minimize,basinhopping,differential_evolution,dual_annealing
from scipy.spatial.distance import cdist
from scipy.special import gammaln


class Optimization_algorithm:
    def __init__(self,opt_method='dual_annealing_multi',fun_name='nmll',**kwargs):
        " The optimization algorithm used for optimizing the hyperparameters "
        self.opt_method=opt_method
        self.fun_name=fun_name
        self.kwargs=kwargs
        self.opt_algo_choise(opt_method)

    def opt_algo_choise(self,opt_method):
        opt_algo={'function':self.function_value,'fixed':self.fixed,'local':self.local,'grid':self.grid,'line':self.line,\
            'random':self.random,'basin':self.basin,'guess_driven':self.guess_driven,'dual_annealing_multi':self.dual_annealing_multi,\
            'dual_annealing_conv':self.dual_annealing_conv,'maximum_est':self.maximum_estimation}
        self.run=opt_algo[opt_method.lower()]
        pass

    def minimizer_method(self,fun,GP,parameters,X_tr,Y_tr,prior=None,jac=True,method='TNC',maxiter=5000,bounds=None,constraints=(),tol=None,options={},dis_m=None,**kwargs):
        " Wrapped scipy local minimizer "
        # The arguments used in the object functions
        args=(GP,parameters,X_tr,Y_tr,prior,jac,dis_m)
        method=method.lower()
        # If jac keyword is also in options
        if 'jac' in options:
            jac=options['jac']
            options.pop('jac')
        # Update options with maxiterations and default values
        options['maxiter']=int(maxiter)
        options_local={'nelder-mead':{'adaptive':True},'l-bfgs-b':{'maxcor':10,'maxls':20},'tnc':{'maxCGit':4,'eta':0.5,'stepmx':5,'accuracy':1e-5,'rescale':3}}
        if method in options_local.keys():
            for key in options_local[method].keys():
                if key not in options:
                    options[key]=options_local[method][key]
        # Update tolerance
        tol_local={'nelder-mead':1e-8,'powell':1e-3,'cg':1e-3,'bfgs':1e-10,'l-bfgs-b':1e-10,'tnc':1e-12,'slsqp':1e-12}
        if tol is None:
            if method in tol_local.keys():
                tol=tol_local[method]
        if method in ['powell','nelder-mead','cobyla']:
            args=list(args)
            args[-2]=False
            args=tuple(args)
            if method=='powell':
                options['maxfev']=int(maxiter)
                min_opt=lambda x0: minimize(fun,x0=x0,method=method,bounds=bounds,tol=tol,options=options,args=args)
            elif method=='nelder-mead':
                options['maxfev']=int(maxiter)
                min_opt=lambda x0: minimize(fun,x0=x0,method=method,tol=tol,options=options,args=args)
            elif method=='cobyla':
                min_opt=lambda x0: minimize(fun,x0=x0,method=method,constraints=constraints,tol=tol,options=options,args=args)
        elif method in ['l-bfgs-b','tnc']:
            if method=='l-bfgs-b':
                options['maxfun']=int(maxiter)
            min_opt=lambda x0: minimize(fun,x0=x0,jac=jac,method=method,bounds=bounds,tol=tol,options=options,args=args)
        elif method=='slsqp':
            min_opt=lambda x0: minimize(fun,x0=x0,jac=jac,method=method,constraints=constraints,tol=tol,options=options,args=args)    
        else:
            min_opt=lambda x0: minimize(fun,x0=x0,jac=jac,method=method,tol=tol,options=options,args=args)
        return min_opt

    def function_value(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds=None,maxiter=2,log=True,prior=None,dis_m=None,options={},jac=False):
        " Calculate the function value for a given set of hyperparameters"
        # Set the calculator up
        sol={'fun':np.inf,'x':theta,'success':True,'nfev':1,'iter':1}
        if jac:
            f_value,f_deriv=fun(theta,GP,parameters,X_tr,Y_tr,prior,jac,dis_m)
            sol['jac']=f_deriv
        else:
            f_value=fun(theta,GP,parameters,X_tr,Y_tr,prior,jac,dis_m)
        sol['fun']=f_value
        return sol

    def fixed(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds=None,maxiter=200,log=True,prior=None,dis_m=None,options={},optimize=False,**kwargs):
        " Calculate the fixed educated guess of a set of hyperparameters"
        # Set the calculator up 
        sol={'fun':fun(theta,GP,parameters,X_tr,Y_tr,prior,False,dis_m),\
                'x':theta,'success':False,'nfev':2,'iter':2}
        from catlearn.regression.gp_bv.educated_guess import Educated_guess
        ed_guess=Educated_guess(GP,fun_name=self.fun_name)
        parameters=list(set(parameters))
        hp=ed_guess.hp(X_tr,Y_tr,parameters=parameters)
        theta=[list(np.array(hp[para]).reshape(-1)) for para in parameters]
        parameters=sum([[para]*len(theta[p]) for p,para in enumerate(parameters)],[])
        theta=sum(theta,[])
        i_sort=np.argsort(parameters)
        theta=np.array(theta)[i_sort]
        parameters=list(np.array(parameters)[i_sort])
        if log:
            theta=np.log(theta)
        f=fun(theta,GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        if f<sol['fun']:
            sol['fun']=f
            sol['x']=theta
            sol['success']=True

        # Local optimize if wanted
        if optimize:
            local_maxiter=int(maxiter-2)
            local_maxiter=0 if local_maxiter<0 else local_maxiter
            min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,prior=prior,maxiter=local_maxiter,dis_m=dis_m,**kwargs)
            mopt=min_opt(sol['x'])
            if mopt['fun']<=sol['fun']:
                sol=mopt
            sol['nfev']+=mopt['nfev']
        return sol

    def local(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds=None,maxiter=5000,log=True,prior=None,dis_m=None,options={},method='TNC',jac=True,constraints=(),tol=None,**kwargs):
        "Local optimization of hyperparameters with scipy minimization"
        # Initial value
        args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        sol={'fun':fun(theta,*args),'x':theta,'success':False}
        i=1
        # Local optimization
        min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,prior=prior,jac=jac,method=method,maxiter=maxiter,bounds=bounds,\
                                        constraints=constraints,tol=tol,options=options,dis_m=dis_m,**kwargs)
        mopt=min_opt(theta)
        if mopt['fun']<=sol['fun']:
            sol=mopt
        i+=mopt['nfev']
        sol['nfev']=i
        return sol
    
    def grid(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=100,log=True,prior=None,dis_m=None,options={},n_each_dim=None,optimize=True,**kwargs):
        "Make a brute force grid optimization of the hyperparameters"
        dim=len(parameters)
        # Make boundary for grid
        if not log:
            bounds=np.log(bounds)

        # Number of points per dimension
        if n_each_dim is None:
            n_each_dim=int(maxiter**(1/len(parameters)))
            n_each_dim=n_each_dim if n_each_dim>1 else 1
        # Make grid either with the same or different numbers in each dimension
        if isinstance(n_each_dim,int):    
            theta_r=[np.linspace(bounds[p][0],bounds[p][1],n_each_dim+2)[1:-1] for p in range(dim)]
        else:
            theta_r=[np.linspace(bounds[p][0],bounds[p][1],n_each_dim[p]+2)[1:-1] for p in range(dim)]
        theta_r=self.make_grid(theta_r,maxiter-1)
        if not log:
            theta_r=np.exp(theta_r)
        theta_r=np.append(theta_r,np.array([theta]),axis=0)

        sol={'fun':np.inf,'success':False}
        # Set the calculator up 
        args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        i=0
        # Calculate the grid points
        for t in theta_r:
            f=fun(t,*args)
            if f<sol['fun']:
                sol['fun']=f
                sol['x']=t
                sol['success']=True
            i+=1
        # Local optimize the best point if wanted
        local_maxiter=int(maxiter-i)
        local_maxiter=0 if local_maxiter<0 else local_maxiter
        min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,maxiter=local_maxiter,prior=prior,dis_m=dis_m,options=options,**kwargs)
        if optimize:
            mopt=min_opt(sol['x'])
            if mopt['fun']<=sol['fun']:
                sol=mopt
            i+=mopt['nfev']
        sol['nfev']=i
        return sol

    def line(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=5000,log=True,options={},prior=None,dis_m=None,n_each_dim=None,loops=3,optimize=True,**kwargs):
        "Make a linesearch in each of the dimensions of the hyperparameters iteratively"
        dim=len(parameters)
        theta=np.array(theta)
        # Make boundary for grid
        if not log:
            bounds=np.log(bounds)

        # Number of points per dimension
        if n_each_dim is None:
            n_each_dim=int(maxiter/(loops*dim))
            n_each_dim=n_each_dim if n_each_dim>1 else 1
        # Make grid either with the same or different numbers in each dimension
        if isinstance(n_each_dim,int):
            n_each_dim=[n_each_dim]*dim
        if sum(n_each_dim)*loops>maxiter:
            n_each_dim=int(maxiter/(loops*dim))
            n_each_dim=n_each_dim if n_each_dim>1 else 1
            n_each_dim=[n_each_dim]*dim
        
        # Set the calculator up 
        args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        sol={'fun':fun(theta,*args),'x':theta,'success':False}
        i=1
        # Calculate the line points
        for l in range(int(loops)):
            for d in range(dim):
                for t in np.linspace(bounds[d][0],bounds[d][1],n_each_dim[d]+2)[1:-1]:
                    theta_r=sol['x'].copy()
                    if log:
                        theta_r[d]=t
                    else:
                        theta_r[d]=np.exp(t)
                    f=fun(theta_r,*args)
                    if f<sol['fun']:
                        sol['fun']=f
                        sol['x']=theta_r.copy()
                        sol['success']=True
                    i+=1

        # Local optimize the best point if wanted
        local_maxiter=int(maxiter-i)
        local_maxiter=0 if local_maxiter<0 else local_maxiter
        min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,maxiter=local_maxiter,prior=prior,dis_m=dis_m,options=options,**kwargs)
        if optimize:
            mopt=min_opt(sol['x'])
            if mopt['fun']<=sol['fun']:
                sol=mopt
            i+=mopt['nfev']
        sol['nfev']=i
        return sol

    def random(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=5000,log=True,prior=None,dis_m=None,npoints=50,jac=True,nopt=None,get_ed_guess=True,dis_min=False,stop_criteria=None,**kwargs):
        " Sample npoints random points in the boundary and local optimize the nopt points with lowest value"
        # Make boundary for random search
        dim=len(parameters)
        if not log:
            bounds=np.log(bounds)
        
        if npoints>maxiter:
            npoints=maxiter
        if nopt is None:
            nopt=npoints

        theta_r=np.array([theta])
        npts_ini=npoints-1
        #Best guess
        if get_ed_guess and npts_ini>0:
            npts_ini=npts_ini-1
            from catlearn.regression.gp_bv.educated_guess import Educated_guess
            ed_guess=Educated_guess(GP,fun_name=self.fun_name)
            parameters_set=sorted(list(set(parameters)))
            hp=ed_guess.hp(X_tr,Y_tr,parameters=parameters_set)
            theta_best=[list(hp[para]) for para in parameters_set]
            theta_best=np.array(sum(theta_best,[]))
            if log:
                theta_best=np.log(theta_best)
            theta_r=np.append(theta_r,[theta_best],axis=0)
            
        # Randomly draw hyperparameters
        if npts_ini>0:
            theta_new=np.random.uniform(low=bounds[:,0],high=bounds[:,1],size=(npts_ini,dim))
            if not log:
                theta_new=np.exp(theta_new)
            theta_r=np.append(theta_r,theta_new,axis=0)

        # Set the calculator up 
        args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)

        # Calculate object functions of all stationary hyperparameters
        f_list=[fun(t,*args) for t in theta_r]
        # Sort after lowest object function values
        sort_min=np.argsort(f_list)[:nopt]

        i=len(f_list)
        sol={'fun':f_list[sort_min[0]],'x':theta_r[sort_min[0]],'success':False,'nfev':i}
        theta_list=theta_r.copy()

        # Local optimize the lowest object function values
        for t in theta_r[sort_min]:
            if i<maxiter:
                local_maxiter=int(maxiter-i)
                local_maxiter=0 if local_maxiter<0 else local_maxiter
                min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,maxiter=local_maxiter,prior=prior,jac=jac,dis_m=dis_m,**kwargs)
            else:
                break
            mopt=min_opt(t)
            if mopt['fun']<=sol['fun']:
                sol=mopt
            i+=mopt['nfev']
            f_list.append(mopt['fun'])
            theta_list=np.append(theta_list,np.array([mopt['x']]),axis=0)
            if stop_criteria is not None:
                if np.sum(np.isclose(f_list[npoints:],sol['fun'],atol=1e-3,rtol=1e-3))>=stop_criteria:
                    if dis_min:
                        if np.sum(np.isclose(cdist(theta_list,np.array([sol['x']])),0,atol=1e-3,rtol=1e-3))>=stop_criteria:
                            sol['success']=True
                            break
                    else:
                        sol['success']=True
                        break

        sol['nfev']=i
        # Fraction of optimal hyperparameters that have same values and are close
        index_min=np.where(np.isclose(f_list,sol['fun'],atol=1e-3,rtol=1e-3))[0]
        if dis_min:
            dis_best=cdist(theta_list[index_min],np.array([sol['x']]))
            sol['Fraction']=np.sum(np.isclose(dis_best,np.min(dis_best),atol=1e-3,rtol=1e-3))/len(f_list)
        else:
            sol['Fraction']=len(index_min)/len(f_list)
        return sol

    def basin(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds=None,maxiter=1000,log=True,prior=None,dis_m=None,options={},method='TNC',niter=5,jac=True,constraints=(),tol=None,interval=10,T=1.0,stepsize=0.1,**kwargs):
        " Basin-hopping optimization of the hyperparameters"
        # Set the calculator up 
        if method in ['powell','nelder-mead','cobyla']:
            args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        else:
            args=(GP,parameters,X_tr,Y_tr,prior,jac,dis_m)
        options['maxiter']=int(maxiter/niter)
        
        method=method.lower()
        # Set the local optimizer parameter
        if method=='slsqp':
            minimizer_kwargs={'method':method,'args':args,'jac':jac,'constraints':constraints,'tol':tol,'options':options}
        elif method in ['l-bfgs-b','tnc']:
            minimizer_kwargs={'method':method,'args':args,'jac':jac,'bounds':bounds,'tol':tol,'options':options}
        elif method=='powell':
            minimizer_kwargs={'method':method,'args':args,'bounds':bounds,'tol':tol,'options':options}
        elif method=='nelder-mead':
            minimizer_kwargs={'method':method,'args':args,'tol':tol,'options':options}
        elif method=='cobyla':
            minimizer_kwargs={'method':method,'args':args,'constraints':constraints,'tol':tol,'options':options}
        else:
            minimizer_kwargs={'method':method,'args':args,'jac':jac,'tol':tol,'options':options}
        
        # Do the basin-hopping
        sol=basinhopping(fun,x0=theta,niter=niter,minimizer_kwargs=minimizer_kwargs,interval=interval,T=T,stepsize=stepsize)
        return sol

    def guess_driven(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=5000,log=True,prior=None,dis_m=None,npoints=4,restart=1,options={},method='TNC',jac=True,get_ed_guess=True,early_stop=True,random_var=0.01,dis_min=False,**kwargs):
        " Educated guess driven approach where the intial set and the best guess of the hyperparameters are used. Furtheremore, pseudo-random points are also used. Restarts are also performed"
        # Make boundary
        dim=len(parameters)
        if not log:
            bounds=np.log(bounds)
        options['maxiter']=int(maxiter)
        # Best guess
        theta_r=[theta]
        npts_ini=npoints-1
        if npts_ini>0:
            if not get_ed_guess:
                theta_r.append(np.mean(bounds,axis=1))
            else:
                from catlearn.regression.gp_bv.educated_guess import Educated_guess
                ed_guess=Educated_guess(GP,fun_name=self.fun_name)
                parameters_set=sorted(list(set(parameters)))
                hp=ed_guess.hp(X_tr,Y_tr,parameters=parameters_set)
                theta=[list(np.array(hp[para]).reshape(-1)) for para in parameters_set]
                theta=np.array(sum(theta,[]))
                if log:
                    theta=np.log(theta)
                theta_r.append(theta)

        theta_r=np.array(theta_r)
        # Pseudo-random sampled sets of hyperparameters
        npts_ini-=1
        if npts_ini>0:
            npts_extra=int(np.ceil((npts_ini+1)**(1/dim)))
            theta_r2=[np.linspace(bounds[p][0],bounds[p][1],npts_extra+2)[1:-1] for p in range(dim)]
            theta_r2=self.make_grid(theta_r2,npts_ini)
            theta_r=np.append(theta_r,theta_r2,axis=0)
        
        if not log:
            if len(theta_r)>1:
                theta_r[1:]=np.exp(theta_r[1:])
        
        args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)

        # Calculate object functions of all stationary hyperparameters
        f_list=[[fun(t,*args) for t in theta_r]]
        theta_list=theta_r.copy()
        i_min=np.argmin(f_list[0])
        sol={'fun':f_list[0][i_min],'x':theta_list[i_min],'success':False}
        nfev=len(f_list[0])
        
        method=method.lower()
        # Set up local optimizer
        min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,method=method,maxiter=maxiter,prior=prior,jac=jac,dis_m=dis_m,**kwargs)
        
        # Run the local optimization with restarts and perturbations
        for r in range(restart+1):
            if nfev<maxiter:
                local_maxiter=int((maxiter-nfev)/len(theta_r))
                local_maxiter=0 if local_maxiter<0 else local_maxiter
                min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,method=method,maxiter=local_maxiter,prior=prior,jac=jac,dis_m=dis_m,**kwargs)
            else:
                break

            sol_list=[min_opt(t) for t in theta_r]
            s_nfev,sol_f,sol_x=[[s[key] for s in sol_list] for key in ['nfev','fun','x']]
            nfev+=np.nansum(s_nfev)
            i_min=np.argmin(sol_f)
            if sol_f[i_min]<=sol['fun']:
                sol=sol_list[i_min] 

            theta_list=np.append(theta_list,sol_x,axis=0)
            f_list.append(sol_f)
            
            if early_stop:
                if np.sum(np.isclose(sol_f,sol['fun'],atol=1e-8))==npoints and npoints>1:
                    if dis_min:
                        if np.sum(np.isclose(cdist(sol_x,np.array([sol['x']])),0,atol=1e-3,rtol=1e-3))==npoints:
                            sol['success']=True
                            break
                    else:
                        sol['success']=True
                        break
            
            if r>0 and r<restart:
                indicies=np.isclose(sol_f,f_list[-2])
                fix_p=sum(indicies)
                if fix_p:
                    rand_c=np.random.uniform(low=1-random_var,high=1+random_var,size=(fix_p,fix_p,1))/fix_p
                    if log:
                        theta_r[indicies]=np.nansum(abs(theta_r[indicies])*rand_c,axis=1)
                    else:
                        theta_r[indicies]=np.exp(np.nansum(np.log(abs(theta_r[indicies]))*rand_c,axis=1))
        
        # Fraction of optimal hyperparameters that have same values and are close
        f_list=np.array(f_list).reshape(-1)
        index_min=np.where(np.isclose(f_list,sol['fun'],atol=1e-3,rtol=1e-3))[0]
        if dis_min:
            dis_best=cdist(theta_list[index_min],np.array([sol['x']]))
            sol['Fraction']=np.sum(np.isclose(dis_best,np.min(dis_best),atol=1e-3,rtol=1e-3))/len(f_list)
        else:
            sol['Fraction']=len(index_min)/len(f_list)
        sol['nit']=(r+1)*npoints
        sol['nfev']=nfev
        return sol

    def dual_annealing_multi(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=5000,log=False,prior=None,dis_m=None,options={},chains=3,initial_temp=5230.0,
                                restart_temp_ratio=2.0e-5,visit=2.62,accept=-5.0,no_local_search=False,miniter_chain=200,append_chains=False,dis_min=False,**kwargs):
        "Remade dual-annealing from scipy that uses chains that are calculated at the same time, where it can be stopped early if the runs are close after a number of runs per chain"
        
        # Make boundary
        dim=len(theta)
        
        # Initial parameters
        TAIL_LIMIT=1e8
        MIN_VISIT_BOUND=1e-10
        lower=bounds[:,0]
        upper=bounds[:,1]
        bound_range=upper-lower
        temperature_restart=initial_temp*restart_temp_ratio
        visit_params=np.sqrt(np.pi)*(np.exp((4-visit)*np.log(visit-1)))/((np.exp((2-visit)*np.log(2)/(visit-1)))*(3-visit))
        factor5=1/(visit-1)-0.5
        factor6=np.pi*(1-factor5)/np.sin(np.pi*(1-factor5))/np.exp(gammaln(2-factor5))
        args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        
        # Default local optimizer
        local_search_options=copy.deepcopy(options)
        if local_search_options=={}:
            local_search_options=copy.deepcopy(local_search_options)
            local_search_options['method']='L-BFGS-B'
            local_search_options['options']={'maxiter':min(max(len(lower)*6,100),1000)}
            local_search_options['bounds']=list(zip(lower,upper))    

        # Random sampled sets of hyperparameters
        theta0=np.array([theta])
        if chains>1:
            theta_r=np.random.uniform(low=lower,high=upper,size=(chains-1,dim))
            theta0=np.append(theta0,theta_r,axis=0)
        theta0=np.array(theta0,dtype=float)
        
        # Initial positions and values
        f0=[fun(theta,*args) for theta in theta0]
        not_improved_idx=np.array([0 for c in range(chains)])
        sol={'fun':f0,'x':theta0,'nfev':0,'niter':0,'success':False,'fun_list':np.array([np.inf]*chains),'x_list':[[]]*chains}
        if append_chains:
            for c in range(chains):
                sol['chain'+str(c)]=np.array([theta0[c]])
                sol['fun'+str(c)]=np.array([fun(theta0[c],*args)])
        x_best=theta0.copy()
        f_best=np.array(f0).copy()
        
        need_to_stop=False
        iteration=0
        t1=np.exp((visit-1)*np.log(2.0))-1.0
        while(not need_to_stop):
            # loop over iterations
            iter_chain=int(maxiter/chains)
            iter_chain=1 if iter_chain<1 else iter_chain
            for i in range(iter_chain):
                t2=np.exp((visit-1)*np.log(float(i+2)))-1
                temperature=initial_temp*t1/t2
                if iteration>=maxiter:
                    need_to_stop=True
                    break
                if temperature<temperature_restart:
                    # restart from maxiter loop to restart temperature
                    theta0=np.random.uniform(low=lower,high=upper,size=(chains,dim))
                    break
                
                not_improved_idx=not_improved_idx+1
                temperature_step=temperature/float(i+1)
                factor4=visit_params*np.exp(np.log(temperature)/(visit-1))
                
                # loop over the chains iteratively
                for c in range(chains):
                    improved=False
                    # One dimension at the time
                    for j in range(dim*2):
                        if iteration>=maxiter:
                            need_to_stop=True
                            break

                        if j==0:
                            if i==0:
                                improved=True
                            else:
                                improved=False
                        
                        theta=theta0[c].copy()
                        if j<dim:
                            x,y=np.random.normal(size=(dim,2)).T
                            x*=np.exp(-(visit-1)*np.log(factor6/factor4)/(3-visit))
                            x_visit=x/np.exp((visit-1)*np.log(np.fabs(y))/(3-visit))
                            upper_sample,lower_sample=np.random.uniform(size=2)
                            x_visit[x_visit>TAIL_LIMIT]=TAIL_LIMIT*upper_sample
                            x_visit[x_visit<-TAIL_LIMIT]=-TAIL_LIMIT*lower_sample
                            theta=theta+x_visit
                            a=theta-lower
                            b=np.fmod(a,bound_range)+bound_range
                            theta=np.fmod(b,bound_range)+lower
                            theta[np.fabs(theta-lower)<MIN_VISIT_BOUND]+=1e-10
                        else:
                            x,y=np.random.normal(size=(1,2)).T
                            x*=np.exp(-(visit-1)*np.log(factor6/factor4)/(3-visit))
                            x_visit=x/np.exp((visit-1)*np.log(np.fabs(y))/(3-visit))
                            if x_visit>TAIL_LIMIT:
                                x_visit=TAIL_LIMIT*np.random.uniform()
                            elif x_visit<-TAIL_LIMIT:
                                x_visit=-TAIL_LIMIT*np.random.uniform()
                            index=j-dim
                            theta[index]=theta[index]+x_visit
                            a=theta[index]-lower[index]
                            b=np.fmod(a,bound_range[index])+bound_range[index]
                            theta[index]=np.fmod(b,bound_range[index])+lower[index]
                            if np.fabs(theta[index]-lower[index])<MIN_VISIT_BOUND:
                                theta[index]+=MIN_VISIT_BOUND
                            
                        f=fun(theta,*args)
                        iteration+=1
                        if f<f0[c]:
                            # Lowest value
                            theta0[c]=theta.copy()
                            f0[c]=f
                            if f<f_best[c]:
                                x_best[c]=theta.copy()
                                f_best[c]=f
                                improved=True
                                not_improved_idx[c]=0
                        else:
                            # Accept/reject
                            pqa=1-(1-accept)*(f-f0[c])/temperature_step
                            if pqa<=0:
                                pqa=0
                            else:
                                pqa=np.exp(np.log(pqa)/(1-accept))               
                            if np.random.uniform()<=pqa:
                                theta0[c]=theta.copy()
                                f0[c]=f
                        
                        if append_chains:
                            sol['chain'+str(c)]=np.append(sol['chain'+str(c)],[theta0[c]],axis=0)
                            sol['fun'+str(c)]=np.append(sol['fun'+str(c)],[f0[c]],axis=0)
                        
                    if iteration>=maxiter:
                        break
                    
                    # Do local optimization
                    if not no_local_search:
                        do_ls=False
                        if improved:
                            do_ls=True  
                        if not_improved_idx[c]>=1000:
                            do_ls=True
                        if do_ls:
                            local_maxiter=int(maxiter-iteration)
                            if 'options' in local_search_options and 'maxiter' in local_search_options['options'] and local_search_options['options']['maxiter']<local_maxiter:
                                local_maxiter=local_search_options['options']['maxiter']
                            min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,maxiter=local_maxiter,prior=prior,dis_m=dis_m,**local_search_options)
                            sol_local=min_opt(x_best[c])

                            iteration+=sol_local['nfev']
                            not_improved_idx[c]=0
                            if sol_local['fun']<f0[c]:
                                theta0[c]=sol_local['x']
                                f0[c]=sol_local['fun']
                                x_best[c]=sol_local['x']
                                f_best[c]=sol_local['fun']
                            if append_chains:
                                sol['chain'+str(c)]=np.append(sol['chain'+str(c)],[sol_local['x']],axis=0)
                                sol['fun'+str(c)]=np.append(sol['fun'+str(c)],[sol_local['fun']],axis=0)
                
                # Stop if all the chains finds the same minimum
                if iteration>miniter_chain*chains and chains>1:
                    i_min=np.argmin(f_best)
                    if np.sum(np.isclose(f_best,f_best[i_min],atol=1e-3,rtol=1e-3))==chains:
                        if dis_min:
                            dis_best=cdist(x_best,np.array([x_best[i_min]]))
                            if np.sum(np.isclose(dis_best,dis_best[i_min],atol=1e-3,rtol=1e-3))==chains:
                                sol['success']=True
                                need_to_stop=True
                                break
                        else:
                            sol['success']=True
                            need_to_stop=True
                            break
                    
        # Ending the optimization   
        sol['x_list']=x_best
        sol['fun_list']=f_best
        i_min=np.argmin(f_best)
        sol['fun']=f_best[i_min]
        sol['x']=x_best[i_min]
        sol['nfev']=iteration
        sol['niter']=int(iteration/chains)
        index_min=np.where(np.isclose(f_best,sol['fun'],atol=1e-3,rtol=1e-3))[0]
        if dis_min:
            dis_best=cdist(x_best[index_min],np.array([sol['x']]))
            sol['Fraction']=np.sum(np.isclose(dis_best,np.min(dis_best),atol=1e-3,rtol=1e-3))/chains
        else:
            sol['Fraction']=len(index_min)/chains
        return sol

    def dual_annealing_conv(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=5000,log=True,prior=None,dis_m=None,options={},dis_min=True,\
                        stop_criteria=400,early_stop=True,initial_temp=5230.0,restart_temp_ratio=2e-05,\
                        visit=2.62,accept=-5.0,seed=None,no_local_search=False,callback=None,args=None,**kwargs):
        " Uses a convergence criteria to stop the dual annelaling "
        dim=len(parameters)

        if args is None:
            args=(GP,parameters,X_tr,Y_tr,prior,False,dis_m)
        
        it_val_t=0
        numb_fmin_t=0
        f_min_t=np.inf
        it_fmin_t=0
        x_min_t=theta.copy()
        success_t=False
        
        def fun_track(theta,GP,parameters,X,Y_tr,prior=None,jac=False,dis_m=None):
            nonlocal it_val_t
            nonlocal f_min_t
            nonlocal it_fmin_t
            nonlocal x_min_t
            nonlocal numb_fmin_t
            nonlocal success_t
            
            if jac:
                f,j=fun(theta,GP,parameters,X,Y_tr,prior=prior,jac=jac,dis_m=dis_m)
            else:
                f=fun(theta,GP,parameters,X,Y_tr,prior=prior,jac=jac,dis_m=dis_m)
            it_val_t+=1
            if f<f_min_t:
                numb_fmin_t=1
                f_min_t=f
                x_min_t=theta.copy()
                it_fmin_t=0
            elif np.isclose(f,f_min_t,atol=1e-3,rtol=1e-3):
                if dis_min:
                    if np.isclose(cdist(np.array([theta]),np.array([x_min_t])),0,atol=1e-3,rtol=1e-3):
                        numb_fmin_t+=1
                else:
                    numb_fmin_t+=1
            it_fmin_t+=1
            if numb_fmin_t>=stop_criteria:
                success_t=True
                if early_stop:
                    raise Exception('The global minima is found!')

            if jac:
                return f,j
            return f
        
        local_search_options=copy.deepcopy(options)
        try:
            sol=dual_annealing(fun_track,bounds,maxfun=maxiter,maxiter=maxiter,args=args,\
                        x0=theta,local_search_options=local_search_options,initial_temp=initial_temp,\
                        restart_temp_ratio=restart_temp_ratio,visit=visit,accept=accept,seed=seed,\
                        no_local_search=no_local_search,callback=callback)
        except:
            sol={'x':x_min_t,'fun':f_min_t,'nfev':it_val_t}
        sol['success']=success_t
        sol['numb_f_min']=numb_fmin_t
        sol['iter_f_min']=it_fmin_t
        return sol

    def maximum_estimation(self,fun,GP,X_tr,Y_tr,theta,parameters,bounds,maxiter=5000,log=True,prior=None,dis_m=None,options={},n_each_dim=100,**kwargs):
        " Make a maximum estimation by first use a line search in one length scale and then a local optimization (Only works with mnlml)"
        # Calculate the minimum noise from the noise correction * 10
        noise_correction=np.sqrt(len(Y_tr)*(1/(1/(4.0*np.finfo(float).eps)-1)))
        GP.set_hyperparams({'noise':np.array([10*noise_correction]),'alpha':np.array([1.0])})
        # Make boundary for grid
        if n_each_dim>maxiter:
            n_each_dim=maxiter
        if not log:
            bounds=np.log(bounds)
        # Make the grid for the length scale
        if 'length' in parameters:
            i_length=parameters.index('length')
            length_grid=np.linspace(bounds[i_length][0],bounds[i_length][1],n_each_dim)[::-1]
        else:
            eps_mach_lower=np.log(10*np.sqrt(2.0*np.finfo(float).eps))
            length_grid=np.linspace(eps_mach_lower,-eps_mach_lower,n_each_dim)[::-1]
        if not log:    
            length_grid=np.exp(length_grid)
        # The line search in the length-scale hyperparameter
        args=(GP,['length'],X_tr,Y_tr,prior,False,dis_m)
        fun_list=[fun(np.array([l]),*args) for l in length_grid]
        i_min=np.nanargmin(fun_list)
        nfev=n_each_dim
        if log:
            GP.set_hyperparams({'length':np.exp([length_grid[i_min]])})
        else:
            GP.set_hyperparams({'length':np.array([length_grid[i_min]])})
        parameters_set=sorted(list(set(parameters)))
        theta_opt=np.array([GP.hp[para][0] for para in parameters_set])
        if log:
            theta_opt=np.log(theta_opt)
        sol={'fun':fun_list[i_min],'success':False,'x':theta_opt}
        # The local optimization in all investigated hyperparameters
        local_maxiter=int(maxiter-nfev)
        local_maxiter=0 if local_maxiter<0 else local_maxiter
        hp={para:sol['x'][p] for p,para in enumerate(parameters_set)}
        theta_opt=np.array([hp[para] for para in parameters])
        min_opt=self.minimizer_method(fun,GP,parameters,X_tr,Y_tr,maxiter=local_maxiter,prior=prior,dis_m=dis_m,options=options,**kwargs)
        mopt=min_opt(theta_opt)
        # Finalize the optimization
        nfev+=mopt['nfev']
        if mopt['fun']<=sol['fun']:
            sol=mopt
        sol['nfev']=nfev
        return sol

    def make_grid(self,lines,maxiter=5000):
        "Make a grid in multi-dimensions from a list of 1D grids in each dimension"
        lines=np.array(lines)
        if len(lines.shape)<2:
            lines=lines.reshape(1,-1)
        #Number of combinations
        combi=1
        for i in [len(line) for line in lines]:
            combi*=i
        if combi<maxiter:
            maxiter=combi
        #If there is a low probability to find grid points randomly the entire grid are calculated
        if (1-(maxiter/combi))<0.99:
            X=lines[0].reshape(-1,1)
            lines=lines[1:]
            for line in lines:
                dim_X=len(X)
                X=np.concatenate([X]*len(line),axis=0)
                X=np.concatenate([X,np.sort(np.concatenate([line.reshape(-1)]*dim_X,axis=0)).reshape(-1,1)],axis=1)
            return np.random.permutation(X)[:maxiter]
        #Randomly sample the grid points
        X=np.array([np.random.choice(line,size=maxiter) for line in lines]).T
        X=np.unique(X,axis=0)
        while len(X)<maxiter:
            x=np.array([np.random.choice(line,size=1) for line in lines]).T
            X=np.append(X,x,axis=0)
            X=np.unique(X,axis=0)
        return X[:maxiter]





