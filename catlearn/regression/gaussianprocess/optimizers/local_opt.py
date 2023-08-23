import numpy as np
from scipy.optimize import minimize,OptimizeResult
from .functions import calculate_list_values,find_minimas

def scipy_opt(fun,x0,jac=True,tol=1e-12,maxiter=5000,args=(),method='L-BFGS-B',options={},**kwargs):
    " Use scipy's minimize to perform a local optimization. "
    if method.lower() in ['nelder-mead']:
        options['maxfev']=int(maxiter)
    elif method.lower() in ['l-bfgs-b','tnc']:
        options['maxfun']=int(maxiter)
    else:
        options['maxiter']=int(maxiter)
    return minimize(fun,x0=x0,method=method,jac=jac,tol=tol,args=tuple(args),options=options,**kwargs)

def golden_search(fun,brack,maxiter=200,tol=1e-3,args=(),fbrack=None,vec0=np.array([0.0]),direc=np.array([1.0]),direc_norm=None,xtol=None,ftol=None,**kwargs):
    " Perform a golden section search. "
    # Set the tolerances criteria
    if xtol is None:
        xtol=tol
    if ftol is None:
        ftol=tol
    # Golden ratio
    r=(np.sqrt(5)-1)/2
    c=1-r
    # Number of function evaluations
    nfev=0
    # Get the coordinates and function values of the endpoints of the interval
    x1,x4=brack
    vec1=vec0+direc*x1
    vec4=vec0+direc*x4
    if fbrack is None:
        f1,f4=fun(vec1,*args),fun(vec4,*args)
        nfev+=2
    else:
        f1,f4=fbrack
    # Direction vector norm
    if direc_norm is None:
        direc_norm=np.linalg.norm(direc)
    # Check if the maximum number of iterations have been used
    if maxiter<3:
        i_min=np.nanargmin([f1,f4])
        sol={'fun':[f1,f4][i_min],'x':[vec1,vec4][i_min],'success':False,'nfev':nfev,'nit':nfev}
        return OptimizeResult(**sol)
    # Check if the convergence criteria is already met in terms of the coordinates
    if abs(x4-x1)*direc_norm<=xtol:
        i_min=np.nanargmin([f1,f4])
        sol={'fun':[f1,f4][i_min],'x':[vec1,vec4][i_min],'success':True,'nfev':nfev,'nit':nfev}
        return OptimizeResult(**sol)
    # Make and calculate points within the interval
    x2,x3=r*x1+c*x4,c*x1+r*x4
    vec2=vec0+direc*x2
    vec3=vec0+direc*x3
    f2=fun(vec2,*args)
    f3=fun(vec3,*args)
    nfev+=2
    # Perform the line search
    success=False
    while nfev<maxiter:
        i_min=np.nanargmin([f1,f2,f3,f4])
        # Check for convergence
        if np.nanmax([f1,f2,f3,f4])-[f1,f2,f3,f4][i_min]<=ftol*(1.0+abs([f1,f2,f3,f4][i_min])) or abs(x4-x1)*direc_norm<=xtol*(1.0+direc_norm*abs(x2)):
            success=True
            break
        # Calculate a new point 
        if i_min<2:
            x4=x3
            f4=f3
            x3=x2
            f3=f2
            x2=r*x3+c*x1
            vec2=vec0+direc*x2
            f2=fun(vec2,*args)
        else:
            x1=x2
            f1=f2
            x2=x3
            f2=f3
            x3=r*x2+c*x4
            vec3=vec0+direc*x3
            f3=fun(vec3,*args)
        nfev+=1
    # Get the solution
    i_min=np.nanargmin([f1,f2,f3,f4])
    sol={'fun':[f1,f2,f3,f4][i_min],'x':vec0+direc*([x1,x2,x3,x4][i_min]),'success':success,'nfev':nfev,'nit':nfev}
    return sol

def run_golden(fun,line,fun_list=calculate_list_values,tol=1e-5,maxiter=5000,optimize=True,multiple_min=True,args=(),fun_kwargs={},theta_index=0,xtol=None,ftol=None,**kwargs):
    " Perform a golden section search as a line search. "
    # Set the tolerances criteria
    if xtol is None:
        xtol=tol
    if ftol is None:
        ftol=tol
    # Calculate function values for line coordinates
    len_l=len(line)
    line=line.reshape(len_l,-1)
    f_list=fun_list(line,fun,*args,**fun_kwargs)
    # Find the optimal value
    i_min=np.nanargmin(f_list)
    # Check whether the object function is flat
    if (np.nanmax(f_list)-f_list[i_min])<8.0e-14:
        i=int(np.floor(0.3*(len(line)-1)))
        return {'fun':f_list[i],'x':line[i],'success':False,'nfev':len_l,'nit':len_l}
    sol={'fun':f_list[i_min],'x':line[i_min],'success':False,'nfev':len_l,'nit':len_l}
    if optimize:
        # Find local minimums or the global minimum
        i_minimas=find_minimas(line,f_list,i_min,len_l,multiple_min=multiple_min,theta_index=theta_index,xtol=xtol,ftol=ftol)
        # Check for convergence 
        len_i=len(i_minimas)
        if len_i==0:
            sol['success']=True
            return sol
        # Do multiple golden section search if necessary
        niter=sol['nfev']
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
            sol_o=golden_search(fun,[0.0,1.0],fbrack=[f1,f4],maxiter=int(maxiter-niter),tol=tol,args=args,vec0=theta0,direc=direc,direc_norm=direc_norm,xtol=xtol,ftol=ftol,**kwargs)
            # Update the solution
            niter+=sol_o['nfev']
            if sol_o['fun']<=sol['fun']:
                sol=sol_o.copy()
            if niter>=maxiter:
                break
        sol['nfev'],sol['nit']=niter,niter
    return sol

def fine_grid_search(fun,line,fun_list=calculate_list_values,sol=None,tol=1e-5,maxiter=5000,loops=5,iterloop=80,optimize=True,multiple_min=True,args=(),fun_kwargs={},lines=[],f_lists=[],theta_index=0,xtol=None,ftol=None,**kwargs):
    " Perform a fine grid search for all minimums. "
    # Set the tolerances criteria
    if xtol is None:
        xtol=tol
    if ftol is None:
        ftol=tol
    # Calculate function values for line coordinates
    len_l=len(line)
    line=line.reshape(len_l,-1)
    f_list=fun_list(line,fun,*args,**fun_kwargs)
    # Use previously calculated grid points
    if len(lines):
        lines=np.append(lines,line,axis=0)
        i_sort=np.argsort(lines[:,theta_index])
        lines=lines[i_sort]
        f_lists=np.append(f_lists,f_list)[i_sort]
    else:
        lines=line.copy()
        f_lists=f_list.copy()
    # Find the minimum value
    i_min=np.nanargmin(f_lists)
    # Make solution dictionary if it does not exist else update it
    if sol is None:
        sol={'fun':f_lists[i_min],'x':lines[i_min],'success':False,'nfev':len_l,'nit':len_l}
    else:
        sol['nfev']+=len_l
        sol['nit']+=len_l
        if f_lists[i_min]<=sol['fun']:
            sol['fun']=f_lists[i_min]
            sol['x']=lines[i_min]
    # Optimize the minimums
    if optimize and loops:
        # Find local minimums or the global minimum
        i_minimas=find_minimas(lines,f_lists,i_min,len(lines),multiple_min=multiple_min,theta_index=theta_index,xtol=xtol,ftol=ftol)
        # Check for convergence
        len_i=len(i_minimas)
        if len_i==0:
            sol['success']=True
            return sol
        # Make a new grid if minimums exist
        if multiple_min:
            if iterloop<len_i*4:
                i_minimas=i_minimas[:iterloop//4]
                len_i=len(i_minimas)
            di=np.full(shape=len_i,fill_value=iterloop//len_i,dtype=int)
            di[:int(iterloop%len_i)]+=1
            newline=np.concatenate([np.linspace(lines[i-1],lines[i+1],di[j]+2)[1:-1] for j,i in enumerate(i_minimas)])
        else:
            i_min=i_minimas[0]
            newline=np.linspace(lines[i_min-1],lines[i_min+1],iterloop+2)[1:-1]
        # Find the grid points that must be saved for later
        i_all=(i_minimas+np.array([[-1],[0],[1]],dtype=int)).T.reshape(-1)
        lines=lines[i_all]
        f_lists=f_lists[i_all]
        return fine_grid_search(fun,newline,fun_list=fun_list,sol=sol,tol=tol,maxiter=int(maxiter-len_l),loops=int(loops-1),iterloop=iterloop,optimize=optimize,multiple_min=multiple_min,args=args,fun_kwargs=fun_kwargs,lines=lines,f_lists=f_lists,theta_index=theta_index,xtol=xtol,ftol=ftol,**kwargs)
    return sol

def fine_grid_search_hptrans(fun,line,fun_list=calculate_list_values,sol=None,tol=1e-5,maxiter=5000,loops=5,iterloop=80,optimize=True,likelihood=False,args=(),fun_kwargs={},lines=[],f_lists=[],theta_index=0,xtol=None,ftol=None,**kwargs):
    " Perform a fine grid search by updating the variable transformation of the hyperparameter. "
    from scipy.integrate import cumulative_trapezoid
    # Set the tolerances criteria
    if xtol is None:
        xtol=tol
    if ftol is None:
        ftol=tol
    # Calculate function values for line coordinates
    len_l=len(line)
    line=line.reshape(len_l,-1)
    f_list=fun_list(line,fun,*args,**fun_kwargs)
    # Use previously calculated grid points
    if len(lines):
        lines=np.append(lines,line,axis=0)
        i_sort=np.argsort(lines[:,theta_index])
        lines=lines[i_sort]
        f_lists=np.append(f_lists,f_list)[i_sort]
    else:
        lines=line.copy()
        f_lists=f_list.copy()
    # Find the minimum value
    i_min=np.nanargmin(f_lists)
    # Make solution dictionary if it does not exist else update it
    if sol is None:
        sol={'fun':f_lists[i_min],'x':lines[i_min],'success':False,'nfev':len_l,'nit':len_l}
    else:
        sol['nfev']+=len_l
        sol['nit']+=len_l
        if f_lists[i_min]<=sol['fun']:
            sol['fun']=f_lists[i_min]
            sol['x']=lines[i_min]
    # Check if the minimum is converged
    i_minimas=find_minimas(lines,f_lists,i_min,len(lines),multiple_min=False,theta_index=theta_index,xtol=xtol,ftol=ftol)
    if len(i_minimas)==0:
        sol['success']=True
        return sol
    # Optimize the minimums
    if optimize and loops:
        # Change the function to likelihood or to a scaled function from 0 to 1
        if likelihood:
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
        newlines=np.linspace(0.0+dl,1.0-dl,iterloop)
        # Find the intervals where the new grid points are located
        i_new=np.where((cdf_r[:-1]<=newlines)&(newlines<cdf_r[1:]))[0]
        i_new_a=i_new+1
        # Calculate the linear interpolation for the intervals of interest
        a=(lines[i_new_a]-lines[i_new])/(cdf_r[i_new_a]-cdf_r[i_new])
        b=((lines[i_new]*cdf_r[i_new_a])-(lines[i_new_a]*cdf_r[i_new]))/(cdf_r[i_new_a]-cdf_r[i_new])
        # Calculate the hyperparameters
        newline=a*newlines.reshape(-1,1)+b
        return fine_grid_search_hptrans(fun,newline,fun_list=fun_list,sol=sol,tol=tol,maxiter=int(maxiter-len_l),loops=int(loops-1),iterloop=iterloop,optimize=optimize,args=args,fun_kwargs=fun_kwargs,lines=lines,f_lists=f_lists,theta_index=theta_index,xtol=xtol,ftol=ftol,**kwargs)
    return sol
