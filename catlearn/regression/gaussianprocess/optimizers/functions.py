import numpy as np

def theta_to_hp(theta,parameters):
    " Transform a list of values and a list of parameter categories to  a dictionary of hyperparameters " 
    return {para_s:theta[np.array(parameters)==para_s] for para_s in sorted(set(parameters))}

def hp_to_theta(hp):
    " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
    parameters_set=sorted(set(hp.keys()))
    theta=[list(np.array(hp[para]).reshape(-1)) for para in parameters_set]
    parameters=sum([[para]*len(theta[p]) for p,para in enumerate(parameters_set)],[])
    theta=np.array(sum(theta,[]))
    return theta,parameters

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

def anneal_var_trans(x,fun,hyper_var,gp,parameters,X,Y,prior=None,jac=False,dis_m=None):
    " Object function called for simulated annealing, where hyperparameter transformation. "
    theta=hp_to_theta(hyper_var.transform_t_to_theta(theta_to_hp(x,parameters)))[0]
    return fun.function(theta,gp,parameters,X,Y,prior,jac,dis_m)

