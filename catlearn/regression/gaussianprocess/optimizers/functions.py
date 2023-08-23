import numpy as np

def theta_to_hp(theta,parameters):
    " Transform a list of values and a list of parameter categories to  a dictionary of hyperparameters " 
    return {para_s:theta[np.array(parameters)==para_s] for para_s in sorted(set(parameters))}

def hp_to_theta(hp):
    " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
    parameters_set=sorted(hp.keys())
    theta=sum([list(hp[para]) for para in parameters_set],[])
    parameters=sum([[para]*len(hp[para]) for para in parameters_set],[])
    return np.array(theta),parameters

def make_grid(lines,maxiter=5000):
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

def make_lines(parameters,model,X,Y,bounds=None,ngrid=80,hptrans=True,use_bounds=True,ngrid_each_dim=False,s=0.14):
    " Make grid in each dimension of the hyperparameters from variable transformation, estimated boundary conditions, or a given boundary conditions. "
    if not ngrid_each_dim:
        ngrid=[ngrid]*len(parameters)
    if bounds is None:
        if hptrans:
            from ..hptrans import Variable_Transformation
            hyper_var=Variable_Transformation().transf_para(parameters,model,X,Y,use_bounds=use_bounds,s=s)
            dl=np.finfo(float).eps
            lines=[np.linspace(0.0+dl,1.0-dl,ngrid[p]) for p in range(len(parameters))]
            lines=hyper_var.t_to_theta_lines(lines,parameters)
        else:
            from ..boundary import Boundary_conditions
            if use_bounds:
                bounds=Boundary_conditions(bound_type='educated',scale=1,max_length=True).create(model,X,Y,parameters,log=True)
            else:
                bounds=Boundary_conditions(bound_type='no',scale=1,max_length=True).create(model,X,Y,parameters,log=True)
            bounds=np.concatenate([bounds[para] for para in sorted(set(parameters))],axis=0)
            lines=[np.linspace(bound[0],bound[1],ngrid[b]) for b,bound in enumerate(bounds)]
    else:
        lines=[np.linspace(bounds[p][0],bounds[p][1],ngrid[p]) for p in range(len(parameters))]
    return lines

def sample_thetas(parameters,model,X,Y,bounds=None,npoints=80,hptrans=True,use_bounds=True,s=0.14):
    " Sample npoints of hyperparameter sets from variable transformation, estimated boundary conditions, or a given boundary conditions. "
    if bounds is None:
        if hptrans:
            from ..hptrans import Variable_Transformation
            hyper_var=Variable_Transformation().transf_para(parameters,model,X,Y,use_bounds=use_bounds,s=s)
            thetas=[]
            for n in range(npoints-1):
                t={para:np.array([np.random.uniform(0.0,1.0)]) for para in parameters}
                hp_t=hyper_var.transform_t_to_theta(t)
                thetas.append(hp_to_theta(hp_t)[0])
            thetas=np.array(thetas)
        else:
            from ..boundary import Boundary_conditions
            if use_bounds:
                bounds=Boundary_conditions(bound_type='educated',scale=1,max_length=True).create(model,X,Y,parameters,log=True)
            else:
                bounds=Boundary_conditions(bound_type='no',scale=1,max_length=True).create(model,X,Y,parameters,log=True)
            bounds=np.concatenate([bounds[para] for para in sorted(set(parameters))],axis=0)
            thetas=np.random.uniform(low=bounds[:,0],high=bounds[:,1],size=(int(npoints-1),len(parameters)))
    else:
        thetas=np.random.uniform(low=bounds[:,0],high=bounds[:,1],size=(int(npoints-1),len(parameters)))
    return thetas

def anneal_var_trans(x,fun,hyper_var,parameters,model,X,Y,pdis=None,jac=False):
    " Object function called for simulated annealing, where hyperparameter transformation. "
    x=np.where(np.where(0.0<x,x,1e-9)<1.0,x,1.00-1e-9)
    theta=hp_to_theta(hyper_var.transform_t_to_theta(theta_to_hp(x,parameters)))[0]
    return fun.function(theta,parameters,model,X,Y,pdis,jac)

def calculate_list_values(line,fun,*args,**kwargs):
    " Calculate a list of values with a function. "
    return np.array([fun(theta,*args,**kwargs) for theta in line])

def find_minimas(xvalues,fvalues,i_min,len_l,multiple_min=True,theta_index=0,xtol=None,ftol=None,**kwargs):
    " Find all the local minimums and their indicies or just the global minimum and then check convergence. "
    # Investigate multiple minimums
    if multiple_min:
        # Find local minimas for middel part of line 
        i_minimas=np.where((fvalues[1:-1]<fvalues[:-2])&(fvalues[2:]>fvalues[1:-1]))[0]+1
        # Check if the function values for the minimums are within the tolerance
        if len(i_minimas):
            i_keep=abs(fvalues[i_minimas+1]+fvalues[i_minimas-1]-2.0*fvalues[i_minimas])>=ftol*(1.0+abs(fvalues[i_minimas]))
            i_minimas=i_minimas[i_keep]
        # Find local minimas for end parts of line
        if fvalues[0]-fvalues[1]<-ftol:
            i_minimas=np.append([1],i_minimas)
        if fvalues[-1]-fvalues[-2]<-ftol:
            i_minimas=np.append(i_minimas,[len_l-2])
        # Check the distances in the local minimas
        if len(i_minimas):
            if theta_index is None:
                i_keep=np.linalg.norm(xvalues[i_minimas+1]-xvalues[i_minimas-1],axis=1)>=xtol*(1.0+np.linalg.norm(xvalues[i_minimas],axis=1))
            else:
                i_keep=abs(xvalues[i_minimas+1,theta_index]-xvalues[i_minimas-1,theta_index])>=xtol*(1.0+abs(xvalues[i_minimas,theta_index]))
            i_minimas=i_minimas[i_keep]
        # Sort the indicies after function value sizes
        if len(i_minimas)>1:
            i_sort=np.argsort(fvalues[i_minimas])
            i_minimas=i_minimas[i_sort]
        return i_minimas
    # Investigate the global minimum
    i_minimas=[]
    if i_min==0:
        # Check if the function values are converged if the endpoints is the minimum value
        if fvalues[0]-fvalues[1]<-ftol:
            i_minimas=[i_min+1]
    elif i_min==int(len_l-1):
        # Check if the function values are converged if the endpoints is the minimum value
        if fvalues[-1]-fvalues[-2]<-ftol:
            i_minimas=[i_min-1]
    else:
        # Check if the function values are converged
        if abs(fvalues[i_min+1]+fvalues[i_min-1]-2.0*fvalues[i_min])>=ftol*(1.0+abs(fvalues[i_min])):
            i_minimas=[i_min]
    # Check if the distance in the local minimum is converged
    if len(i_minimas):
        i_minima=i_minimas[0]
        if not np.linalg.norm(xvalues[i_minima+1]-xvalues[i_minima-1])>=xtol*(1.0+np.linalg.norm(xvalues[i_minima])):
            i_minimas=[]
    return np.array(i_minimas)
