import numpy as np
import copy
from scipy.spatial.distance import cdist,pdist,squareform


class Kernel:
    def __init__(self,hp={'length':np.array([0.5]),'alpha':np.array([1.0])}):
        """The Kernel class with hyperparameters.
            Parameters:
                hp : Hyperparameters
                    A dictionary of hyperparameters like length scale, and prefactor
        """
        self.hp=hp
        self.set_hyperparams(hp)

    def dist_m(self,X,Q=None):
        """Calculate the squared distance matrix
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
        """
        if Q is None:
            return pdist(X,metric='sqeuclidean')
        return cdist(X,Q,metric='sqeuclidean')


    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        return self.hp
    
    def __repr__(self):
        return 'Kernel(hp={})'.format(self.hp)



    
class SE(Kernel):
    """ The Squared exponential kernel matrix (or Radial-basis function kernel) with one length scale without derivatives of target. """

    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'length' not in self.hp:
            self.hp['length']=np.array([0.5])
        if 'alpha' not in self.hp:
            self.hp['alpha']=np.array([1.0])

        #Lower and upper machine precision
        eps_mach_lower=np.sqrt(2.0*np.finfo(float).eps)
        eps_mach_upper=1/eps_mach_lower

        self.hp['length']=np.abs(self.hp['length']).reshape(-1)
        self.hp['alpha']=np.abs(self.hp['alpha']).reshape(-1)
        self.hp['length']=np.where(self.hp['length']<eps_mach_upper,np.where(self.hp['length']>eps_mach_lower,self.hp['length'],eps_mach_lower),eps_mach_upper)
        self.hp['alpha']=np.where(self.hp['alpha']<eps_mach_upper,np.where(self.hp['alpha']>eps_mach_lower,self.hp['alpha'],eps_mach_lower),eps_mach_upper)
        return self.hp

    def dist_m(self,X,Q=None):
        """Calculate the distance matrix divided by the length scale
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
        """
        if Q is None:
            return pdist(X,metric='sqeuclidean')
        return cdist(X,Q,metric='sqeuclidean')

    def __call__(self,X,Q=None,get_derivatives=True,dists=None):
        """Make the kernel matrix.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
                get_derivatives : bool
                    Can only be False.
                dists : (N,M) or (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it.
        """
        if dists is None:
            if Q is None:
                dists=self.dist_m(X/self.hp['length'])
            else:
                dists=self.dist_m(X/self.hp['length'],Q/self.hp['length'])
        else:
            dists=dists/(self.hp['length']**2)
        if Q is None:
            K=(self.hp['alpha']**2)*np.exp(-0.5*dists)
            K=squareform(K)
            np.fill_diagonal(K,self.hp['alpha']**2)
        else:
            K=(self.hp['alpha']**2)*np.exp(-0.5*dists)
        return K

    def diag(self,X,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
        """
        return np.array([self.hp['alpha']**2]*len(X)).reshape(-1)

    def get_derivatives(self,X,hp,KXX=None,dists=None):
        """Get the derivatives of the kernel matrix in respect to the hyperparameters.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array (optional)
                    The kernel matrix of training data .
                dists : (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it 
        """
        hp_deriv={}
        distsl=None
        if 'alpha' in hp:
            if KXX is None:
                if dists is None:
                    distsl=self.dist_m(X/self.hp['length'])
                else:
                    distsl=dists/(self.hp['length']**2)
                Kd=(2*self.hp['alpha'])*np.exp(-0.5*distsl)
                Kd=squareform(Kd)
                np.fill_diagonal(Kd,2*self.hp['alpha'])
            else:
                Kd=(2/self.hp['alpha'])*KXX
            hp_deriv['alpha']=Kd
        if 'length' in hp:
            if distsl is None:
                if dists is None:
                    distsl=self.dist_m(X/self.hp['length'])
                else:
                    distsl=dists/(self.hp['length']**2)
            if KXX is None:
                Kd=((self.hp['alpha']**2)*(distsl/self.hp['length']))*np.exp(-0.5*distsl)
                Kd=squareform(Kd)
            else:
                Kd=squareform(distsl/self.hp['length'])*KXX
            hp_deriv['length']=Kd
        return hp_deriv

    def __repr__(self):
        return 'SE(hp={})'.format(self.hp)



class SE_Deriv(Kernel):
    """ The Squared exponential kernel matrix (or Radial-basis function kernel) with one length scale with derivatives/gradients of target. """

    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'length' not in self.hp:
            self.hp['length']=np.array([0.5])
        if 'alpha' not in self.hp:
            self.hp['alpha']=np.array([1.0])

        #Lower and upper machine precision
        eps_mach_lower=np.sqrt(2.0*np.finfo(float).eps)
        eps_mach_upper=1/eps_mach_lower

        self.hp['length']=np.abs(self.hp['length']).reshape(-1)
        self.hp['alpha']=np.abs(self.hp['alpha']).reshape(-1)
        self.hp['length']=np.where(self.hp['length']<eps_mach_upper,np.where(self.hp['length']>eps_mach_lower,self.hp['length'],eps_mach_lower),eps_mach_upper)
        self.hp['alpha']=np.where(self.hp['alpha']<eps_mach_upper,np.where(self.hp['alpha']>eps_mach_lower,self.hp['alpha'],eps_mach_lower),eps_mach_upper)
        return self.hp

    def dist_m(self,X,Q=None):
        """Calculate the distance tensor
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
        """
        if Q is None:
            return np.array([X[:,d:d+1]-X[:,d] for d in range(len(X[0]))])
        return np.array([X[:,d:d+1]-Q[:,d] for d in range(len(X[0]))])

    def __call__(self,X,Q=None,get_derivatives=True,dists=None):
        """Make the kernel matrix.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
                get_derivatives : bool
                    Can only be False.
                dists : (N,M) or (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it.
        """
        dim=len(X[0])
        if dists is None:
            dists=self.dist_m(X,Q)
        
        if Q is None:
            nd1=len(X)
            Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
            K=(self.hp['alpha']**2)*np.exp((-0.5/(self.hp['length']**2))*np.sum(dists**2,axis=0))
            dists=dists/(self.hp['length']**2)
            Kext[:nd1,:nd1]=K
            for d1 in range(1,dim+1):
                dis_K=(-K)*dists[d1-1]
                Kext[:nd1,nd1*d1:nd1*(d1+1)]=-dis_K
                Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K
                Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=K/(self.hp['length']**2)+dists[d1-1]*dis_K
                for d2 in range(d1+1,dim+1):
                    Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=dists[d2-1]*dis_K
        else:
            nd1=len(X) ; nd2=len(Q)
            if get_derivatives:
                Kext=np.zeros((nd1*(dim+1),nd2*(dim+1)))
            else:
                Kext=np.zeros((nd1,nd2*(dim+1)))
            K=(self.hp['alpha']**2)*np.exp((-0.5/(self.hp['length']**2))*np.sum(dists**2,axis=0))
            dists=dists/(self.hp['length']**2)
            Kext[:nd1,:nd2]=K
            for d1 in range(1,dim+1):
                dis_K=(-K)*dists[d1-1]
                Kext[:nd1,nd2*d1:nd2*(d1+1)]=-dis_K
                if get_derivatives:
                    Kext[nd1*d1:nd1*(d1+1),:nd2]=dis_K
                    Kext[nd1*d1:nd1*(d1+1),nd2*d1:nd2*(d1+1)]=K/(self.hp['length']**2)+(dists[d1-1]*dis_K)
                    for d2 in range(d1+1,dim+1):
                        Kext[nd1*d1:nd1*(d1+1),nd2*d2:nd2*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd2*d1:nd2*(d1+1)]=dists[d2-1]*dis_K
        return Kext

    def diag(self,X,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
        """
        if get_derivatives:
            return np.array([self.hp['alpha']**2]*len(X)+[self.hp['alpha']**2/(self.hp['length']**2)]*len(X)*len(X[0])).reshape(-1)
        return np.array([self.hp['alpha']**2]*len(X)).reshape(-1)


    def get_derivatives(self,X,hp,KXX=None,dists=None):
        """Get the derivatives of the kernel matrix in respect to the hyperparameters.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array (optional)
                    The kernel matrix of training data .
                dists : (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it 
        """
        hp_deriv={}
        dim=len(X[0])
        distsl=None
        if 'alpha' in hp:
            if KXX is None:
                if dists is None:
                    dists=self.dist_m(X)
                nd1=len(X)
                Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
                K=(2*self.hp['alpha'])*np.exp((-0.5/(self.hp['length']**2))*np.sum(dists**2,axis=0))
                distsl=dists/(self.hp['length']**2)
                Kext[:nd1,:nd1]=K
                for d1 in range(1,dim+1):
                    dis_K=(-K)*distsl[d1-1]
                    Kext[:nd1,nd1*d1:nd1*(d1+1)]=-dis_K
                    Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K
                    Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=K/(self.hp['length']**2)+distsl[d1-1]*dis_K
                    for d2 in range(d1+1,dim+1):
                        Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=distsl[d2-1]*dis_K
            else:
                Kext=(2/self.hp['alpha'])*KXX
            hp_deriv['alpha']=Kext
        if 'length' in hp:
            if dists is None:
                dists=self.dist_m(X)
            distl_s=np.sum(dists**2,axis=0)/(self.hp['length']**2)
            if KXX is None: 
                K=(self.hp['alpha']**2)*np.exp(-0.5*distl_s)
                nd1=len(X)
                Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
                if distsl is None:
                    distsl=dists/(self.hp['length']**2)
                distl_s=distl_s/self.hp['length']
                Kext[:nd1,:nd1]=K*distl_s
                for d1 in range(1,dim+1):
                    dis_K=K*distsl[d1-1]
                    Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K*(2/self.hp['length']-distl_s)
                    Kext[:nd1,nd1*d1:nd1*(d1+1)]=-Kext[nd1*d1:nd1*(d1+1),:nd1]
                    disdis_K=distsl[d1-1]*dis_K
                    Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=(K/(self.hp['length']**2)-disdis_K)*distl_s-K*(2/(self.hp['length']**3))+(4/self.hp['length'])*disdis_K
                    for d2 in range(d1+1,dim+1):
                        Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=((4/self.hp['length'])-distl_s)*distsl[d2-1]*dis_K
            else:
                nd1=len(X)
                Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
                if distsl is None:
                    distsl=dists/(self.hp['length']**2)
                K=KXX[:nd1,:nd1].copy()
                distl_s=distl_s/self.hp['length']
                Kext[:nd1,:nd1]=KXX[:nd1,:nd1]*distl_s
                for d1 in range(1,dim+1):
                    Kext[nd1*d1:nd1*(d1+1),:nd1]=KXX[:nd1,nd1*d1:nd1*(d1+1)]*(2/self.hp['length']-distl_s)
                    Kext[:nd1,nd1*d1:nd1*(d1+1)]=-Kext[nd1*d1:nd1*(d1+1),:nd1]
                    dis_K=distsl[d1-1]*K
                    Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]*distl_s+(4/self.hp['length'])*distsl[d1-1]*dis_K-(2/self.hp['length']**3)*K
                    for d2 in range(d1+1,dim+1):
                        Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]*distl_s+(4/self.hp['length'])*distsl[d2-1]*dis_K
            hp_deriv['length']=Kext
        return hp_deriv

    def __repr__(self):
        return 'SE_Deriv(hp={})'.format(self.hp)



class SE_Multi(Kernel):
    """ The Squared exponential kernel matrix (or Radial-basis function kernel) with multiple length scales without derivatives of target. """

    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'length' not in self.hp:
            self.hp['length']=np.array([0.5])
        if 'alpha' not in self.hp:
            self.hp['alpha']=np.array([1.0])

        #Lower and upper machine precision
        eps_mach_lower=np.sqrt(2.0*np.finfo(float).eps)
        eps_mach_upper=1/eps_mach_lower

        self.hp['length']=np.abs(self.hp['length']).reshape(-1)
        self.hp['alpha']=np.abs(self.hp['alpha']).reshape(-1)
        self.hp['length']=np.where(self.hp['length']<eps_mach_upper,np.where(self.hp['length']>eps_mach_lower,self.hp['length'],eps_mach_lower),eps_mach_upper)
        self.hp['alpha']=np.where(self.hp['alpha']<eps_mach_upper,np.where(self.hp['alpha']>eps_mach_lower,self.hp['alpha'],eps_mach_lower),eps_mach_upper)
        return self.hp

    def dist_m(self,X,Q=None):
        """Calculate the distance matrix divided by the length scale
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
        """
        if Q is None:
            return np.array([pdist(X[:,d:d+1],metric='sqeuclidean') for d in range(len(X[0]))])
        return np.array([(X[:,d:d+1]-Q[:,d])**2 for d in range(len(X[0]))])

    def __call__(self,X,Q=None,get_derivatives=True,dists=None):
        """Make the kernel matrix.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
                get_derivatives : bool
                    Can only be False.
                dists : (N,M) or (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it.
        """
        if len(self.hp['length'])!=len(X[0]):
            self.hp['length']=np.array([self.hp['length'].item(0)]*len(X[0]))
        if dists is None:
            if Q is None:
                dists=self.dist_m(X/self.hp['length'])
            else:
                dists=self.dist_m(X/self.hp['length'],Q/self.hp['length'])
        else:
            dists=dists/(self.hp['length'].reshape(-1,1)**2)
        dists=np.sum(dists,axis=0)
        if Q is None:
            K=(self.hp['alpha']**2)*np.exp(-0.5*dists)
            K=squareform(K)
            np.fill_diagonal(K,self.hp['alpha']**2)
        else:
            K=(self.hp['alpha']**2)*np.exp(-0.5*dists)
        return K

    def diag(self,X,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
        """
        return np.array([self.hp['alpha']**2]*len(X)).reshape(-1)


    def get_derivatives(self,X,hp,KXX=None,dists=None):
        """Get the derivatives of the kernel matrix in respect to the hyperparameters.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list with elements of the hyperparameters that are optimized.
                KXX : (N,N) array (optional)
                    The kernel matrix of training data.
                dists : (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it.
        """
        hp_deriv={}
        distsl=None
        distsl_sum=None
        if 'alpha' in hp:
            if KXX is None:
                if dists is None:
                    distsl=self.dist_m(X/self.hp['length'])
                else:
                    distsl=dists/(self.hp['length'].reshape(-1,1)**2)
                distsl_sum=np.sum(distsl,axis=0)
                Kd=(2*self.hp['alpha'])*np.exp(-0.5*distsl_sum)
                Kd=squareform(Kd)
                np.fill_diagonal(Kd,2*self.hp['alpha'])
            else:
                Kd=(2/self.hp['alpha'])*KXX
            hp_deriv['alpha']=Kd
        if 'length' in hp:
            if distsl is None:
                if dists is None:
                    distsl=self.dist_m(X/self.hp['length'])
                else:
                    distsl=dists/(self.hp['length'].reshape(-1,1)**2)
            if distsl_sum is None:
                distsl_sum=np.sum(distsl,axis=0)
            if KXX is None:
                Kd=(self.hp['alpha']**2)*np.exp(-0.5*distsl_sum)
                Kd=np.array([squareform((distsl[d]/self.hp['length'][d])*Kd) for d in range(len(X[0]))])
            else:
                Kd=np.array([squareform(distsl[d]/self.hp['length'][d])*KXX for d in range(len(X[0]))])
            if len(X[0])>1:
                hp_deriv['length']=Kd
            else:
                hp_deriv['length']=Kd[0]
        return hp_deriv

    def __repr__(self):
        return 'SE_Multi(hp={})'.format(self.hp)



class SE_Multi_Deriv(Kernel):
    """ The Squared exponential kernel matrix (or Radial-basis function kernel) with multi length scale with derivatives/gradients of target. """

    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'length' not in self.hp:
            self.hp['length']=np.array([0.5])
        if 'alpha' not in self.hp:
            self.hp['alpha']=np.array([1.0])

        #Lower and upper machine precision
        eps_mach_lower=np.sqrt(2.0*np.finfo(float).eps)
        eps_mach_upper=1/eps_mach_lower

        self.hp['length']=np.abs(self.hp['length']).reshape(-1)
        self.hp['alpha']=np.abs(self.hp['alpha']).reshape(-1)
        self.hp['length']=np.where(self.hp['length']<eps_mach_upper,np.where(self.hp['length']>eps_mach_lower,self.hp['length'],eps_mach_lower),eps_mach_upper)
        self.hp['alpha']=np.where(self.hp['alpha']<eps_mach_upper,np.where(self.hp['alpha']>eps_mach_lower,self.hp['alpha'],eps_mach_lower),eps_mach_upper)
        return self.hp

    def dist_m(self,X,Q=None):
        """Calculate the distance tensor
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
        """
        if Q is None:
            return np.array([X[:,d:d+1]-X[:,d] for d in range(len(X[0]))])
        return np.array([X[:,d:d+1]-Q[:,d] for d in range(len(X[0]))])

    def __call__(self,X,Q=None,get_derivatives=True,dists=None):
        """Make the kernel matrix.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                Q : (M,D) array
                    Features with M data points and D dimensions. If it is not given a squared kernel from X is generated.
                get_derivatives : bool
                    Can only be False.
                dists : (N,M) or (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it.
        """
        dim=len(X[0])
        if len(self.hp['length'])!=dim:
            self.hp['length']=np.array([self.hp['length'].item(0)]*dim)
        if dists is None:
            dists=self.dist_m(X,Q)
        
        if Q is None:
            nd1=len(X)
            Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
            K=(self.hp['alpha']**2)*np.exp((-0.5)*np.sum((dists/self.hp['length'].reshape(-1,1,1))**2,axis=0))
            dists=dists/(self.hp['length'].reshape(-1,1,1)**2)
            Kext[:nd1,:nd1]=K
            for d1 in range(1,dim+1):
                dis_K=(-K)*dists[d1-1]
                Kext[:nd1,nd1*d1:nd1*(d1+1)]=-dis_K
                Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K
                Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=K/(self.hp['length'][d1-1]**2)+dists[d1-1]*dis_K
                for d2 in range(d1+1,dim+1):
                    Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=dists[d2-1]*dis_K
        else:
            nd1=len(X) ; nd2=len(Q)
            if get_derivatives:
                Kext=np.zeros((nd1*(dim+1),nd2*(dim+1)))
            else:
                Kext=np.zeros((nd1,nd2*(dim+1)))
            K=(self.hp['alpha']**2)*np.exp((-0.5)*np.sum((dists/self.hp['length'].reshape(-1,1,1))**2,axis=0))
            dists=dists/(self.hp['length'].reshape(-1,1,1)**2)
            Kext[:nd1,:nd2]=K
            for d1 in range(1,dim+1):
                dis_K=(-K)*dists[d1-1]
                Kext[:nd1,nd2*d1:nd2*(d1+1)]=-dis_K
                if get_derivatives:
                    Kext[nd1*d1:nd1*(d1+1),:nd2]=dis_K
                    Kext[nd1*d1:nd1*(d1+1),nd2*d1:nd2*(d1+1)]=K/(self.hp['length'][d1-1]**2)+(dists[d1-1]*dis_K)
                    for d2 in range(d1+1,dim+1):
                        Kext[nd1*d1:nd1*(d1+1),nd2*d2:nd2*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd2*d1:nd2*(d1+1)]=dists[d2-1]*dis_K
        return Kext

    def diag(self,X,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
        """
        if get_derivatives:
            return np.array(list(np.array([self.hp['alpha']**2]*len(X)).reshape(-1))+list((np.array([self.hp['alpha']**2]*len(X))/(self.hp['length']**2)).reshape(-1))).reshape(-1)
        return np.array([self.hp['alpha']**2]*len(X)).reshape(-1)


    def get_derivatives(self,X,hp,KXX=None,dists=None):
        """Get the derivatives of the kernel matrix in respect to the hyperparameters.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array (optional)
                    The kernel matrix of training data .
                dists : (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it 
        """
        hp_deriv={}
        dim=len(X[0])
        distsl=None
        if 'alpha' in hp:
            if KXX is None:
                if dists is None:
                    dists=self.dist_m(X)
                nd1=len(X)
                Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
                K=(2*self.hp['alpha'])*np.exp((-0.5)*np.sum((dists/self.hp['length'].reshape(-1,1,1))**2,axis=0))
                distsl=dists/(self.hp['length'].reshape(-1,1,1)**2)
                Kext[:nd1,:nd1]=K
                for d1 in range(1,dim+1):
                    dis_K=(-K)*distsl[d1-1]
                    Kext[:nd1,nd1*d1:nd1*(d1+1)]=-dis_K
                    Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K
                    Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=K/(self.hp['length'][d1-1]**2)+distsl[d1-1]*dis_K
                    for d2 in range(d1+1,dim+1):
                        Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=distsl[d2-1]*dis_K
            else:
                Kext=(2/self.hp['alpha'])*KXX
            hp_deriv['alpha']=Kext
        if 'length' in hp:
            if dists is None:
                dists=self.dist_m(X)
            distl_s=(dists/self.hp['length'].reshape(-1,1,1))**2
            if KXX is None: 
                distl_s=(dists/self.hp['length'].reshape(-1,1,1))**2
                K=(self.hp['alpha']**2)*np.exp(-0.5*np.sum(distl_s,axis=0))
                nd1=len(X)
                if distsl is None:
                    distsl=dists/(self.hp['length'].reshape(-1,1,1)**2)
                hp_deriv['length']=[]
                for d in range(dim):
                    Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
                    distl_s2=distl_s[d]/self.hp['length'][d]
                    Kext[:nd1,:nd1]=K*distl_s2[d]
                    for d1 in range(1,dim+1):
                        dis_K=K*distsl[d1-1]
                        if d==d1-1:
                            Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K*(2/self.hp['length'][d1-1]-distl_s2)
                        else:
                            Kext[nd1*d1:nd1*(d1+1),:nd1]=dis_K*(-distl_s2)    
                        Kext[:nd1,nd1*d1:nd1*(d1+1)]=-Kext[nd1*d1:nd1*(d1+1),:nd1]
                        disdis_K=distsl[d1-1]*dis_K
                        if d==d1-1:
                            Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=(K/(self.hp['length'][d1-1]**2)-disdis_K)*distl_s2-K*(2/(self.hp['length'][d1-1]**3))+(4/self.hp['length'][d1-1])*disdis_K
                        else:
                            Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=(K/(self.hp['length'][d1-1]**2)-disdis_K)*distl_s2
                        for d2 in range(d1+1,dim+1):
                            if d==d1-1:
                                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=((2/self.hp['length'][d1-1])-distl_s2)*distsl[d2-1]*dis_K
                            elif d==d2-1:
                                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=((2/self.hp['length'][d2-1])-distl_s2)*distsl[d2-1]*dis_K
                            else:
                                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=(-distl_s2)*distsl[d2-1]*dis_K
                    hp_deriv['length'].append(Kext)
            else:
                nd1=len(X)
                if distsl is None:
                    distsl=dists/(self.hp['length'].reshape(-1,1,1)**2)
                K=KXX[:nd1,:nd1].copy()
                hp_deriv['length']=[]
                for d in range(dim):
                    Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
                    distl_s=dists[d]**2/(self.hp['length'][d]**3)
                    Kext[:nd1,:nd1]=KXX[:nd1,:nd1]*distl_s
                    for d1 in range(1,dim+1):
                        if d==d1-1:
                            Kext[nd1*d1:nd1*(d1+1),:nd1]=KXX[:nd1,nd1*d1:nd1*(d1+1)]*(2/self.hp['length'][d1-1]-distl_s)
                        else:
                            Kext[nd1*d1:nd1*(d1+1),:nd1]=KXX[:nd1,nd1*d1:nd1*(d1+1)]*(-distl_s)
                        Kext[:nd1,nd1*d1:nd1*(d1+1)]=-Kext[nd1*d1:nd1*(d1+1),:nd1]
                        dis_K=distsl[d1-1]*K
                        if d==d1-1:
                            Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]*distl_s+(4/self.hp['length'][d1-1])*distsl[d1-1]*dis_K-(2/self.hp['length'][d1-1]**3)*K
                        else:
                            Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]*distl_s
                        for d2 in range(d1+1,dim+1):
                            if d==d1-1:
                                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]*distl_s+(2/self.hp['length'][d1-1])*distsl[d2-1]*dis_K
                            elif d==d2-1:
                                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]*distl_s+(2/self.hp['length'][d2-1])*distsl[d2-1]*dis_K
                            else:
                                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]*distl_s
                    hp_deriv['length'].append(Kext)
            if dim>1:
                hp_deriv['length']=np.array(hp_deriv['length'])
            else:
                hp_deriv['length']=np.array(hp_deriv['length'][0])
        return hp_deriv

    def __repr__(self):
        return 'SE_Multi_Deriv(hp={})'.format(self.hp)

