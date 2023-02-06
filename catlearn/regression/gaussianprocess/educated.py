import numpy as np
import copy
from scipy.spatial.distance import pdist,squareform
from .fingerprint.fingerprintobject import FingerprintObject

class Educated_guess:
    def __init__(self,GP=None):
        "Educated guess method for hyperparameters of a Gaussian Process"
        if GP is None:
            from .gp.gp import GaussianProcess
            GP=GaussianProcess()
        self.GP=copy.deepcopy(GP)

    def hp(self,X,Y,parameters=None):
        " Get the best educated guess of the hyperparameters "
        if parameters is None:
            parameters=list(self.GP.hp.keys())
            parameters=parameters+['noise']
        if 'correction' in parameters:
            parameters.remove('correction')
        parameters=sorted(parameters)
        hp={}
        for para in sorted(set(parameters)):
            if para=='prefactor':
                hp['prefactor']=np.array(self.prefactor_mean(X,Y)).reshape(-1)
            elif para=='length':
                hp['length']=np.array(self.length_mean(X,Y)).reshape(-1)
            elif para=='noise':
                if 'noise_deriv' in parameters:
                    hp['noise']=np.array(self.noise_mean(X,Y[:,0:1])).reshape(-1)
                else:
                    hp['noise']=np.array(self.noise_mean(X,Y)).reshape(-1)
            elif para=='noise_deriv':
                hp['noise_deriv']=np.array(self.noise_mean(X,Y[:,1:])).reshape(-1)
            else:
                hp[para]=self.no_guess_mean(para,parameters)
        return hp

    def bounds(self,X,Y,parameters=None,scale=1):
        " Get the educated guess bounds of the hyperparameters "
        if parameters is None:
            parameters=list(self.GP.hp.keys())
            parameters=parameters+['noise']
        if 'correction' in parameters:
            parameters.remove('correction')
        parameters=sorted(parameters)
        bounds={}
        for para in sorted(set(parameters)):
            if para=='prefactor':
                bounds['prefactor']=np.array(self.prefactor_bound(X,Y,scale=scale)).reshape(-1,2)
            elif para=='length':
                bounds['length']=np.array(self.length_bound(X,Y,scale=scale)).reshape(-1,2)
            elif para=='noise':
                if 'noise_deriv' in parameters:
                    bounds[para]=np.array(self.noise_bound(X,Y[:,0:1],scale=scale)).reshape(-1,2)
                else:
                    bounds[para]=np.array(self.noise_bound(X,Y,scale=scale)).reshape(-1,2)
            elif para=='noise_deriv':
                bounds[para]=np.array(self.noise_bound(X,Y[:,1:],scale=scale)).reshape(-1,2)
            else:
                bounds[para]=self.no_guess_bound(para,parameters)
        return bounds

    def no_guess_mean(self,para,parameters):
        " Best guess if the parameters is not known. "
        return np.array([0.0]*parameters.count(para))

    def no_guess_bound(self,para,parameters):
        " Bounds if the parameters is not known. "
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        return np.array([[eps_mach_lower,1/eps_mach_lower]]*parameters.count(para))

    def prefactor_mean(self,X,Y):
        "The best educated guess for the prefactor by using standard deviation of the target"
        self.GP.prior.update(X,Y)
        a_mean=np.sqrt(np.mean(((Y[:,0]-self.GP.prior.get(X)[:,0]))**2))
        if a_mean==0.0:
            return 0.00
        return np.log(a_mean)

    def prefactor_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the prefactor in the educated guess regime within a scale"
        a_mean=self.prefactor_mean(X,Y)
        return np.array([a_mean-np.log(scale*10),a_mean+np.log(scale*10)])

    def noise_mean(self,X,Y):
        "The best educated guess for the noise by using the minimum and maximum eigenvalues"
        return np.log(1e-4)

    def noise_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the noise in the educated guess regime within a scale"
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        n_max=len(Y.reshape(-1))
        return np.log([eps_mach_lower,n_max])
    
    def length_mean(self,X,Y):
        "The best educated guess for the length scale by using nearst neighbor"
        lengths=[]
        l_dim=self.GP.kernel.get_dimension(X)
        if isinstance(X[0],FingerprintObject):
            X=np.array([fp.get_vector() for fp in X])
        for d in range(l_dim):
            if l_dim==1:
                dis=pdist(X)
            else:
                dis=pdist(X[:,d:d+1])
            dis=np.where(dis==0.0,np.nan,dis)
            if len(dis)==0:
                dis=[1.0]
            dis_min,dis_max=0.2*np.nanmedian(self.nearest_neighbors(dis)),np.nanmedian(dis)*4.0
            if self.GP.use_derivatives:
                dis_min=dis_min*0.05
            lengths.append(np.nanmean(np.log([dis_min,dis_max])))
        return np.array(lengths)

    def length_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the length scale in the educated guess regime within a scale"
        lengths=[]
        l_dim=self.GP.kernel.get_dimension(X)
        if isinstance(X[0],FingerprintObject):
            X=np.array([fp.get_vector() for fp in X])
        for d in range(l_dim):
            if l_dim==1:
                dis=pdist(X)
            else:
                dis=pdist(X[:,d:d+1])
            dis=np.where(dis==0.0,np.nan,dis)
            if len(dis)==0:
                dis=[1.0]
            dis_min,dis_max=0.2*np.nanmedian(self.nearest_neighbors(dis)),np.nanmedian(dis)*4.0
            if self.GP.use_derivatives:
                dis_min=dis_min*0.05
            lengths.append([dis_min/scale,dis_max*scale])
        return np.log(lengths)
    
    def nearest_neighbors(self,dis):
        " Nearst neighbor distance "
        dis_matrix=squareform(dis)
        m_len=len(dis_matrix)
        dis_matrix[range(m_len),range(m_len)]=np.inf
        return np.nanmin(dis_matrix,axis=0)

