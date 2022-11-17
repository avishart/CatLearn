import numpy as np
from scipy.spatial.distance import pdist,squareform
from .fingerprint.fingerprint import Fingerprint

class Boundary_conditions:
    def __init__(self,bound_type='no',scale=1,max_length=True):
        " Different types of boundary conditions "
        self.bound_type=bound_type
        self.scale=scale
        self.max_length=max_length

    def create(self,GP,X,Y,parameters,log=True):
        " Create the boundary condition from the parameters given "
        if self.bound_type is None:
            return None
        self.bound_type=self.bound_type.lower()
        if self.bound_type=='no':
            bounds=self.create_no(parameters)
        elif self.bound_type=='length':
            bounds=self.create_length(GP,X,Y,parameters)
        elif self.bound_type=='restricted':
            bounds=self.create_restricted(GP,X,Y,parameters)
        elif self.bound_type=='educated':
            bounds=self.create_educated(GP,X,Y,parameters)
        if log:
            return {key:np.log(value) for key,value in bounds.items()}
        return bounds

    def create_no(self,parameters):
        " Create the boundary condition, where no information is known "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        return {para:np.array([[eps_mach_lower,1/eps_mach_lower]]*parameters.count(para)) for para in sorted(set(parameters))}

    def create_length(self,GP,X,Y,parameters):
        " Create the boundary condition, where it is known that the length must be larger than a value "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        bounds={}
        for para in sorted(set(parameters)):
            if para=='length':
                bounds[para]=self.length_bound(X,Y,GP,scale=self.scale)
            else:
                bounds[para]=np.array([[eps_mach_lower,1/eps_mach_lower]]*parameters.count(para))
        return bounds

    def create_restricted(self,GP,X,Y,parameters):
        " Create the boundary condition, where it is known that the length must be larger than a value and a large noise is not favorable for regression "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        bounds={}
        for para in sorted(set(parameters)):
            if para=='length':
                bounds[para]=self.length_bound(X,Y,GP,scale=self.scale)
            elif para=='noise':
                bounds[para]=self.noise_bound(X,Y,scale=self.scale)
            else:
                bounds[para]=np.array([[eps_mach_lower,1/eps_mach_lower]]*parameters.count(para))
        return bounds

    def create_educated(self,GP,X,Y,parameters):
        " Use educated guess for making the boundary conditions "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        bounds={}
        for para in sorted(set(parameters)):
            if para=='length':
                bounds[para]=self.length_bound(X,Y,GP,scale=self.scale)
            elif para=='noise':
                if 'noise_deriv' in parameters and GP.use_derivatives:
                    bounds[para]=self.noise_bound(X,Y[:,0:1],scale=self.scale)
                else:
                    bounds[para]=self.noise_bound(X,Y,scale=self.scale)
            elif para=='noise_deriv' and GP.use_derivatives:
                bounds[para]=self.noise_bound(X,Y[:,1:],scale=self.scale)
            elif para=='prefactor':
                bounds[para]=self.prefactor_bound(X,Y,GP,scale=self.scale)
            else:
                bounds[para]=np.array([[eps_mach_lower,1/eps_mach_lower]]*parameters.count(para))
        return bounds

    def length_bound(self,X,Y,GP,scale=1):
        "Get the minimum and maximum ranges of the length scale in the educated guess regime within a scale"
        exp_lower,exp_max=np.sqrt(-1/np.log(np.finfo(float).eps)),np.sqrt(-1/np.log(1-np.finfo(float).eps))
        if not self.max_length:
            exp_max=2.0
        lengths=[]
        l_dim=GP.kernel.get_dimension(X)
        if isinstance(X[0],Fingerprint):
            X=np.array([fp.get_vector() for fp in X])
        for d in range(l_dim):
            if l_dim==1:
                dis=pdist(X)
                exp_max=2.0
            else:
                dis=pdist(X[:,d:d+1])
            dis=np.where(dis==0.0,np.nan,dis)
            if len(dis)==0:
                dis=[1.0]
            dis_min,dis_max=np.nanmedian(self.nearest_neighbors(dis))*exp_lower,np.nanmax(dis)*exp_max
            lengths.append([dis_min/scale,dis_max*scale])
        return np.array(lengths)

    def noise_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the noise in the educated guess regime within a scale"
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        n_max=len(Y.reshape(-1))
        return np.array([[eps_mach_lower,n_max*scale]])

    def prefactor_bound(self,X,Y,GP,scale=1):
        "Get the minimum and maximum ranges of the prefactor in the educated guess regime within a scale"
        GP.prior.update(X,Y)
        a_mean=np.sqrt(np.mean(((Y[:,0]-GP.prior.get(X)[:,0]))**2))
        if a_mean==0.0:
            return 1.00
        return np.array([[a_mean/(scale*10),a_mean*(scale*10)]])
    
    def nearest_neighbors(self,dis):
        " Nearst neighbor distance "
        dis_matrix=squareform(dis)
        m_len=len(dis_matrix)
        dis_matrix[range(m_len),range(m_len)]=np.inf
        return np.nanmin(dis_matrix,axis=0)
        