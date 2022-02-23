
import numpy as np
from scipy.spatial.distance import pdist


class Boundary_conditions:
    def __init__(self,bound_type='no',scale=1):
        " Different types of boundary conditions "
        self.bound_type=bound_type
        self.scale=scale

    def create(self,GP,X,Y,parameters,log,fun_name):
        " Create the boundary condition from the parameters given "
        if self.bound_type is None:
            return None
        self.bound_type=self.bound_type.lower()
        if self.bound_type=='no':
            bounds=self.create_no(parameters)
        elif self.bound_type=='length':
            bounds=self.create_length(GP,X,Y,parameters,fun_name)
        elif self.bound_type=='restricted':
            bounds=self.create_restricted(GP,X,Y,parameters,fun_name)
        elif self.bound_type=='educated':
            bounds=self.create_educated(GP,X,Y,parameters,fun_name)
        if log:
            return np.log(bounds)
        return bounds

    def create_no(self,parameters):
        " Create the boundary condition, where no information is known "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        return np.array([[eps_mach_lower,1/eps_mach_lower]]*len(parameters))

    def create_length(self,GP,X,Y,parameters,fun_name):
        " Create the boundary condition, where it is known that the length must be larger than a value "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        l_count=len(X[0]) if 'Multi' in str(GP.kernel) else 1
        parameters_set=sorted(list(set(parameters)))
        bounds=[]
        for para in parameters_set:
            if para=='length':
                if 'SE' in str(GP.kernel):
                    exp_lower,exp_max=-1/np.log(np.finfo(float).eps),-1/np.log(1-np.finfo(float).eps)
                    if l_count>1:
                        dist2=np.array([pdist(X[:,d].reshape(-1,1),metric='sqeuclidean') for d in range(l_count)])
                        dist2=np.array([dis[dis!=0] for dis in dist2])
                        for d in range(l_count):
                            if len(dist2[d])>0:
                                bounds.append([np.sqrt(0.5*np.min(dist2[d])*exp_lower),np.sqrt(0.5*np.max(dist2[d])*exp_max)])
                            else:
                                bounds.append([eps_mach_lower,1/eps_mach_lower])
                    else:
                        dist2=pdist(X,metric='sqeuclidean')
                        dist2=dist2[dist2!=0]
                        if len(dist2)>0:
                            bounds.append([np.sqrt(0.5*np.min(dist2)*exp_lower),np.sqrt(0.5*np.max(dist2)*exp_max)])
                        else:
                            bounds.append([eps_mach_lower,1/eps_mach_lower])
                else:
                    for d in range(l_count):
                        bounds.append([eps_mach_lower,1/eps_mach_lower])
            else:
                bounds.append([eps_mach_lower,1/eps_mach_lower])
        bounds=np.array(bounds)
        bounds[:,0]=np.where(bounds[:,0]>=eps_mach_lower,bounds[:,0],eps_mach_lower)
        bounds[:,1]=np.where(bounds[:,1]<=1/eps_mach_lower,bounds[:,1],1/eps_mach_lower)
        return bounds

    def create_restricted(self,GP,X,Y,parameters,fun_name):
        " Create the boundary condition, where it is known that the length must be larger than a value and a large noise is not favorable for regression "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        l_count=len(X[0]) if 'Multi' in str(GP.kernel) else 1
        parameters_set=sorted(list(set(parameters)))
        bounds=[]
        for para in parameters_set:
            if para=='length':
                if 'SE' in str(GP.kernel):
                    exp_lower,exp_max=-1/np.log(np.finfo(float).eps),-1/np.log(1-np.finfo(float).eps)
                    if l_count>1:
                        dist2=np.array([pdist(X[:,d].reshape(-1,1),metric='sqeuclidean') for d in range(l_count)])
                        dist2=np.array([dis[dis!=0] for dis in dist2])
                        for d in range(l_count):
                            if len(dist2[d])>0:
                                bounds.append([np.sqrt(0.5*np.min(dist2[d])*exp_lower),np.sqrt(0.5*np.max(dist2[d])*exp_max)])
                            else:
                                bounds.append([eps_mach_lower,1/eps_mach_lower])
                    else:
                        dist2=pdist(X,metric='sqeuclidean')
                        dist2=dist2[dist2!=0]
                        if len(dist2)>0:
                            bounds.append([np.sqrt(0.5*np.min(dist2)*exp_lower),np.sqrt(0.5*np.max(dist2)*exp_max)])
                        else:
                            bounds.append([eps_mach_lower,1/eps_mach_lower])
                else:
                    for d in range(l_count):
                        bounds.append([eps_mach_lower,1/eps_mach_lower])
            elif para=='noise':
                if fun_name in ['mnll','mnlp']:
                    bounds.append([eps_mach_lower,1.0])
                else:
                    GP.prior.update(X,Y[:,0])
                    Y_std=np.sqrt(np.mean((Y[:,0]-GP.prior.get(X))**2))
                    if Y_std==0.0:
                        Y_std=1.0
                    bounds.append([eps_mach_lower,0.25*Y_std])
            else:
                bounds.append([eps_mach_lower,1/eps_mach_lower])
        bounds=np.array(bounds)
        bounds[:,0]=np.where(bounds[:,0]>=eps_mach_lower,bounds[:,0],eps_mach_lower)
        bounds[:,1]=np.where(bounds[:,1]<=1/eps_mach_lower,bounds[:,1],1/eps_mach_lower)
        return bounds

    def create_educated(self,GP,X,Y,parameters,fun_name):
        " Use educated guess for making the boundary conditions "
        from catlearn.regression.gp_bv.educated_guess import Educated_guess
        ed_guess=Educated_guess(GP,fun_name=fun_name)
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        l_count=parameters.count('length')
        parameters_set=sorted(list(set(parameters)))
        bounds=[]
        for para in parameters_set:
            if para=='alpha':
                bounds.append(ed_guess.alpha_bound(X,Y,scale=self.scale))
            elif para=='length':
                l_bound=ed_guess.length_bound(X,Y,scale=self.scale)
                if l_count>1:
                    for d in range(l_count):
                        bounds.append(l_bound[d])
                else:
                    bounds.append(l_bound)
            elif para=='noise':
                bounds.append(ed_guess.noise_bound(X,Y,scale=self.scale))
            else:
                bounds.append([eps_mach_lower,1/eps_mach_lower])
        bounds=np.array(bounds)
        bounds[:,0]=np.where(bounds[:,0]>=eps_mach_lower,bounds[:,0],eps_mach_lower)
        bounds[:,1]=np.where(bounds[:,1]<=1/eps_mach_lower,bounds[:,1],1/eps_mach_lower)
        return bounds

