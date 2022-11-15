import numpy as np
from scipy.optimize import minimize,OptimizeResult

def scipy_opt(fun,x0,jac=True,tol=1e-12,maxiter=5000,args=(),method='L-BFGS-B',options={},**kwargs):
    " Use scipy's minimize to perform a local optimization "
    options['maxiter']=int(maxiter)
    return minimize(fun,x0=x0,method=method,jac=jac,tol=tol,args=tuple(args),options=options,**kwargs)

def golden_search(fun,brack,maxiter=200,tol=0.001,args=(),fbrack=None,vec0=np.array([0]),direc=np.array([1]),**kwargs):
    'Perform a golden section search '
    #Golden ratio
    r=(np.sqrt(5)-1)/2
    c=1-r
    sol={'success':False,'nfev':0}
    # Bracket
    x1,x4=brack
    if fbrack is None:
        f1,f4=fun(vec0+direc*x1,*args),fun(vec0+direc*x4,*args)
        sol['nfev']+=2
    else:
        f1,f4=fbrack
    # Make points in bracket
    x2,x3=r*x1+c*x4,c*x1+r*x4
    if maxiter<1:
        i_min=np.nanargmin([f1,f4])
        sol['fun']=[f1,f4][i_min]
        sol['x']=vec0+direc*([x1,x4][i_min])
        return OptimizeResult(**sol)
    if abs(x3-x2)==0:
        f2=fun(vec0+direc*x2,*args)
        i_min=np.nanargmin([f1,f2,f4])
        sol['fun'],sol['success']=[f1,f2,f4][i_min],True
        sol['x']=vec0+direc*([x1,x2,x4][i_min])
        sol['nfev']+=1
        return OptimizeResult(**sol)
    f2=fun(vec0+direc*x2,*args)
    f3=fun(vec0+direc*x3,*args)
    sol['nfev']+=2
    # Perform the line search
    while sol['nfev']<maxiter:
        if f1==f2==f3==f4:
            break
        i_min=np.nanargmin([f1,f2,f3,f4])
        if i_min<2:
            x4=x3 ; f4=f3
            x3=x2 ; f3=f2
            x2=r*x3+c*x1
            f2=fun(vec0+direc*x2,*args)
        else:
            x1=x2 ; f1=f2
            x2=x3 ; f2=f3
            x3=r*x2+c*x4
            f3=fun(vec0+direc*x3,*args)
        sol['nfev']+=1
        if tol*(x2+x3)>abs(x4-x1):
            sol['success']=True
            break
    i_min=np.nanargmin([f1,f2,f3,f4])
    sol['fun']=[f1,f2,f3,f4][i_min]
    sol['x']=vec0+direc*([x1,x2,x3,x4][i_min])
    return OptimizeResult(**sol)

def run_golden(fun,line,tol=1e-5,maxiter=5000,optimize=True,multiple_max=True,args=(),**kwargs):
    " Perform a golden section search as a line search "
    # Calculate function values for line coordinates
    len_l=len(line)
    f_list=np.array([fun(theta,*args) for theta in line])
    # Find the optimal value
    i_min=np.nanargmin(f_list)
    # Check whether the object function is flat
    if (np.nanmax(f_list)-f_list[i_min])<8.0e-14:
        i=int(np.floor(0.3*(len(line)-1)))
        return {'fun':f_list[i],'x':line[i],'nfev':len_l,'success':False}
    sol={'fun':f_list[i_min],'x':line[i_min],'nfev':len_l,'success':False}
    if optimize:
        # Find local minimas
        i_minimas=np.where((f_list[1:-1]<f_list[:-2])&(f_list[2:]>f_list[1:-1]))[0]+1
        if f_list[0]<f_list[1] or len(i_minimas)==0:
            i_minimas=np.append([0],i_minimas)
        if f_list[-1]<f_list[-2]:
            i_minimas=np.append(i_minimas,[len_l-1])
        # Do multiple golden section search if necessary
        niter=sol['nfev']
        if multiple_max:
            i_sort=np.argsort(f_list[i_minimas])
        else:
            i_sort=np.array([np.argmin(f_list[i_minimas])])
        for i_min in i_minimas[i_sort]:
            x1=i_min-1 if i_min-1>=0 else i_min
            x4=i_min+1 if i_min+1<=len_l-1 else i_min
            f1,f4=f_list[x1],f_list[x4]
            theta0=line[x1].copy()
            direc=line[x4]-theta0
            sol_o=golden_search(fun,[0.0,1.0],fbrack=[f1,f4],maxiter=int(maxiter-niter),tol=tol,args=args,vec0=theta0,direc=direc,**kwargs)
            niter+=sol_o['nfev']
            if sol_o['fun']<=sol['fun']:
                sol=sol_o.copy()
            if niter>=maxiter:
                break
        sol['nfev']=niter
    return sol

