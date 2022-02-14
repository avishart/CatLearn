
import numpy as np
import copy
from scipy.spatial.distance import pdist,squareform


class Educated_guess:
    def __init__(self,GP=None,fun_name='nmll'):
        "Educated guess method for hyperparameters of a Gaussian Process"
        if GP is None:
            from catlearn.regression.gp_bv.gp import GaussianProcess
            GP=GaussianProcess()
        self.GP=copy.deepcopy(GP)
        self.kernel=copy.deepcopy(GP.kernel)
        self.kernel_type=str(self.kernel).split('(')[0]
        self.fun_name=fun_name


    def hp(self,X,Y,parameters=None):
        " Get the best educated guess of the hyperparameters "
        if parameters is None:
            parameters=list(self.kernel.hp.keys())
            parameters=parameters+['noise']
        parameters=sorted(parameters)
        if 'correction' in parameters:
            parameters.remove('correction')
        hp={}
        for para in sorted(list(set(parameters))):
            if para=='alpha':
                hp['alpha']=np.array(self.alpha_mean(X,Y)).reshape(-1)
                self.kernel.set_hyperparams(hp)
            elif para=='length':
                hp['length']=np.array(self.length_mean(X,Y)).reshape(-1)
            elif para=='noise':
                hp['noise']=np.array(self.noise_mean(X,Y)).reshape(-1)
        return hp

    def alpha_mean(self,X,Y):
        "The best educated guess for the prefactor by using standard deviation of the target"
        if self.kernel_type in ['SE','SE_Deriv','SE_Multi','SE_Multi_Deriv']:
            self.GP.prior.update(X,Y[:,0])
            a_mean=np.sqrt(np.mean((Y[:,0]-self.GP.prior.get(X))**2))
        else:
            self.GP.prior.update(X,Y[:,0])
            a_mean=np.sqrt(np.mean((Y[:,0]-self.GP.prior.get(X))**2))
        if a_mean==0.0:
            return 1.00
        return a_mean

    def alpha_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the prefactor in the educated guess regime within a scale"
        a_mean=self.alpha_mean(X,Y)
        return np.array([a_mean/(scale*10),a_mean*(scale*10)])

    def noise_mean(self,X,Y):
        "The best educated guess for the noise by using the minimum and maximum eigenvalues"
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        a_mean=0.25*self.alpha_mean(X,Y)
        if self.fun_name.lower() in ['mnll','mnlp']:
            a_mean=1.00
        return 10**(0.5*(np.log10(eps_mach_lower)+np.log10(a_mean)))

    def noise_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the noise in the educated guess regime within a scale"
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        a_mean=0.25*self.alpha_mean(X,Y)
        if self.fun_name.lower() in ['mnll','mnlp']:
            a_mean=1.00
        return np.array([eps_mach_lower,a_mean])

    def length_mean(self,X,Y):
        "The best educated guess for the length scale by using nearst neighbor"
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        eps_mach_upper=1/eps_mach_lower
        if self.kernel_type in ['SE','SE_Deriv']:
            dis=squareform(pdist(X))
            dis[range(len(dis)),range(len(dis))]=np.max(dis)
            dis=np.min(dis,axis=1)
            dis=dis[dis!=0]
            if len(dis)>0:
                dis_min,dis_max=0.5*np.min(dis),4*np.max(dis)
            else:
                dis_min,dis_max=eps_mach_lower,eps_mach_upper
            dis_min=np.where(dis_min<eps_mach_upper,np.where(dis_min>eps_mach_lower,dis_min,eps_mach_lower),eps_mach_upper)
            dis_max=np.where(dis_max<eps_mach_upper,np.where(dis_max>eps_mach_lower,dis_max,eps_mach_lower),eps_mach_upper)
            return np.array([10**(0.5*(np.log10(dis_min)+np.log10(dis_max)))])
        elif self.kernel_type in ['SE_Multi','SE_Multi_Deriv']:
            length=[]
            for d in range(len(X[0])):
                dis=squareform(pdist(X[:,d].reshape(-1,1)))
                dis[range(len(dis)),range(len(dis))]=np.max(dis)
                dis=np.min(dis,axis=1)
                dis=dis[dis!=0]
                if len(dis)>0:
                    dis_min,dis_max=0.5*np.min(dis),4*np.max(dis)
                else:
                    dis_min,dis_max=eps_mach_lower,eps_mach_upper
                dis_min=np.where(dis_min<eps_mach_upper,np.where(dis_min>eps_mach_lower,dis_min,eps_mach_lower),eps_mach_upper)
                dis_max=np.where(dis_max<eps_mach_upper,np.where(dis_max>eps_mach_lower,dis_max,eps_mach_lower),eps_mach_upper)
                length.append(10**(0.5*(np.log10(dis_min)+np.log10(dis_max))))
            return np.array(length)
        if 'Multi' in self.kernel_type:
            return np.array([0.5]*len(X[0]))
        return np.array([0.5])

    def length_bound(self,X,Y,scale=1):
        "Get the minimum and maximum ranges of the length scale in the educated guess regime within a scale"
        eps_mach_lower=10*np.sqrt(2.0*np.finfo(float).eps)
        eps_mach_upper=1/eps_mach_lower
        if self.kernel_type in ['SE','SE_Deriv']:
            dis=squareform(pdist(X))
            dis[range(len(dis)),range(len(dis))]=np.max(dis)
            dis=np.min(dis,axis=1)
            dis=dis[dis!=0]
            if len(dis)>0:
                dis_min,dis_max=0.5*np.min(dis)/scale,4*np.max(dis)*scale
            else:
                dis_min,dis_max=eps_mach_lower,eps_mach_upper
            dis_min=np.where(dis_min<eps_mach_upper,np.where(dis_min>eps_mach_lower,dis_min,eps_mach_lower),eps_mach_upper)
            dis_max=np.where(dis_max<eps_mach_upper,np.where(dis_max>eps_mach_lower,dis_max,eps_mach_lower),eps_mach_upper)
            return np.array([dis_min,dis_max])
        elif self.kernel_type in ['SE_Multi','SE_Multi_Deriv']:
            exp_lower,exp_max=-1/np.log(np.finfo(float).eps),-1/np.log(1-np.finfo(float).eps)
            dim=len(X[0])
            dist2=np.array([pdist(X[:,d].reshape(-1,1),metric='sqeuclidean') for d in range(dim)])
            dist2=np.array([dis[dis!=0] for dis in dist2])
            return np.array([[np.sqrt(0.5*np.min(dis)*exp_lower),np.sqrt(0.5*np.max(dis)*exp_max)] if len(dis)>0 else [eps_mach_lower,eps_mach_upper] for dis in dist2])
        if 'Multi' in self.kernel_type:
            return np.array([[eps_mach_lower,eps_mach_upper]]*len(X[0]))
        return np.array([eps_mach_lower,eps_mach_upper])





