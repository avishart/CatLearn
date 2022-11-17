import numpy as np
from scipy.spatial.distance import squareform
from .kernel import Kernel,Kernel_Derivative

class SE(Kernel):
    """ Squared exponential or radial basis kernel """
    
    def get_K(self,features,features2=None,dis_m=None,**kwargs):
        """Make the kernel matrix.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points and D dimensions. 
                    If it is not given a squared kernel from features is generated.
                dis_m : (N,M) or (N*(N-1)/2) array (optional)
                    Already calculated distance matrix.
        """
        if dis_m is None:
            D=self.distances(features,features2,scale=np.exp(-self.hp['length']))
        else:
            D=dis_m*np.exp(-2*self.hp['length'])
        K=np.exp(-0.5*D)
        if features2 is None:
            K=squareform(K)
            np.fill_diagonal(K,1.0)
        return K
    
    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'length' not in self.hp:
            self.hp['length']=np.array([-0.7])
        self.hp['length']=np.array(self.hp['length'],dtype=float).reshape(-1)
        return self.hp

    def diag(self,features,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
        """
        return np.array([1.0]*len(features))
        
    def get_gradients(self,features,hp,KXX,dis_m=None,**kwargs):
        """Get the gradients of the kernel matrix in respect to the hyperparameters.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array
                    The kernel matrix of training data .
                dis_m : (N,M) or (N*(N-1)/2) array (optional)
                    Already calculated distance matrix.
        """
        hp_deriv={}
        if 'length' in hp:
            if dis_m is None:
                D=self.distances(features,scale=np.exp(-self.hp['length']))
            else:
                D=dis_m*np.exp(-2*self.hp['length'])
            hp_deriv['length']=squareform(D)*KXX
        return hp_deriv
    
    def get_dimension(self,features):
        " Get the dimension of the length-scale hyperparameter "
        return 1
    
    def __repr__(self):
        return 'SE(hp={})'.format(self.hp)
    


class SE_Derivative(Kernel_Derivative):
    """ Squared exponential or radial basis kernel with derivatives wrt. to fingerprints """
    
    def get_K(self,dis_m,**kwargs):
        """Make the kernel matrix without derivatives.
            Parameters:
                dis_m : (N,M) or (N*(N-1)/2) array
                    Already calculated distance matrix.
        """
        D=self.distances.get_absolute_distances(dis_m)*np.exp(-2*self.hp['length'])
        K=np.exp(-0.5*D)
        return K
    
    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        if 'length' not in self.hp:
            self.hp['length']=np.array([-0.7])
        self.hp['length']=np.array(self.hp['length'],dtype=float).reshape(-1)
        return self.hp
    
    def get_derivative_K(self,features,dis_m,K,d1,axis=0):
        """Make the derivative of the kernel matrix wrt. to one dimension of the fingerprint.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                dis_m : (D,N,M) array 
                    Already calculated distance matrix.
                K : (N,M) array
                    The kernel matrix without derivatives
                d1 : int
                    The dimension considered.
                axis : int
                    If it is the first or second term in the distance matrix.
        """
        return K*(-0.5*np.exp(-2*self.hp['length']))*self.distances.get_derivative(features,dis_m,d1-1,axis)
    
    def get_hessian_K1(self,features,features2,dis_m,K,d1,d2):
        """Make the first part hessian of the kernel matrix wrt. to two dimension of the fingerprint.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points.
                dis_m : (D,N,M) array 
                    Already calculated distance matrix.
                K : (N,M) array
                    The kernel matrix without derivatives
                d1 : int
                    The dimension considered for features.
                d2 : int
                    The dimension considered for features2.
        """
        dis_d1=self.distances.get_derivative(features,dis_m,d1-1)
        dis_d2=self.distances.get_derivative(features2,dis_m,d2-1,axis=1)
        Kdd1=K*(0.25*np.exp(-4*self.hp['length']))*dis_d1*dis_d2
        return Kdd1
    
    def get_hessian_K2(self,features,features2,dis_m,K,d1,d2):
        """Make the second part hessian of the kernel matrix wrt. to two dimension of the fingerprint.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points.
                dis_m : (D,N,M) array 
                    Already calculated distance matrix.
                K : (N,M) array
                    The kernel matrix without derivatives
                d1 : int
                    The dimension considered for features.
                d2 : int
                    The dimension considered for features2.
        """
        dis_h=self.distances.get_hessian(features,features2,dis_m,d1-1,d2-1)
        if isinstance(dis_h,int):
            return None
        return K*((-0.5*np.exp(-2*self.hp['length']))*dis_h)
    
    def get_hessian_K(self,features,features2,dis_m,K,d1,d2):
        """Make the hessian of the kernel matrix wrt. to two dimension of the fingerprint.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points.
                dis_m : (D,N,M) array 
                    Already calculated distance matrix.
                K : (N,M) array
                    The kernel matrix without derivatives
                d1 : int
                    The dimension considered for features.
                d2 : int
                    The dimension considered for features2.
        """
        Kdd1=self.get_hessian_K1(features,features2,dis_m,K,d1,d2)
        Kdd2=self.get_hessian_K2(features,features2,dis_m,K,d1,d2)
        if Kdd2 is None:
            return Kdd1
        return Kdd1+Kdd2
    
    def get_KXX(self,features,dis_m=None,**kwargs):
        """ Get the symmetric kernel matrix. 
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                dis_m : (D,N,M) array (optional)
                    Already calculated distance matrix.
        """
        # Get dimensions
        dim=self.distances.get_derivative_dimension(features)
        nd1=len(features)
        # Calculate the distances, distance matrix, and kernel matrix
        if dis_m is None:
            dis_m=self.distances(features)
        K=self.get_K(dis_m,**kwargs)
        # Calculate the full symmetric kernel matrix
        Kext=np.zeros((nd1*(dim+1),nd1*(dim+1)))
        Kext[:nd1,:nd1]=K.copy()
        for d1 in range(1,dim+1):
            Kext[nd1*d1:nd1*(d1+1),:nd1]=self.get_derivative_K(features,dis_m,K,d1,axis=0)
            Kext[:nd1,nd1*d1:nd1*(d1+1)]=Kext[nd1*d1:nd1*(d1+1),:nd1].copy().T
            for d2 in range(d1,dim+1):
                Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=self.get_hessian_K(features,features,dis_m,K,d1,d2)
                if d1!=d2:
                    Kext[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=Kext[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)].copy().T
        return Kext
    
    def get_KQX(self,features,features2=None,get_derivatives=False,dis_m=None,**kwargs):
        """ Get the kernel matrix with two different features.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points.
                get_derivatives : bool
                    Whether to get the derivatives of the prediction part.
                dis_m : (D,N,M) array (optional)
                    Already calculated distance matrix.
        """
        # Get dimensions
        dim=self.distances.get_derivative_dimension(features)
        nd1=len(features)
        # Calculate the distances, distance matrix, and kernel matrix
        if dis_m is None:
            dis_m=self.distances(features,features2)
        K=self.get_K(dis_m,**kwargs)
        nd2=len(features2)        
        # Calculate the kernel matrix
        if get_derivatives:
            Kext=np.zeros((nd1*(dim+1),nd2*(dim+1)))
        else:
            Kext=np.zeros((nd1,nd2*(dim+1)))
        Kext[:nd1,:nd2]=K.copy()
        for d1 in range(1,dim+1):
            Kext[:nd1,nd2*d1:nd2*(d1+1)]=self.get_derivative_K(features2,dis_m,K,d1,axis=1)
            if get_derivatives:
                Kext[nd1*d1:nd1*(d1+1),:nd2]=self.get_derivative_K(features,dis_m,K,d1,axis=0)
                for d2 in range(d1,dim+1):
                    Kext[nd1*d1:nd1*(d1+1),nd2*d2:nd2*(d2+1)]=self.get_hessian_K(features,features2,dis_m,K,d1,d2)
                    if d1!=d2:
                        Kext[nd1*d2:nd1*(d2+1),nd2*d1:nd2*(d1+1)]=self.get_hessian_K(features,features2,dis_m,K,d2,d1)
        return Kext

    def diag(self,features,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                get_derivatives : bool
                    Whether to get the derivatives of the prediction part.
        """
        k_diag=np.array([1.0]*len(features))
        if get_derivatives:
            k_hes_diag=(-0.5*np.exp(-2*self.hp['length']).item(0))*self.distances.get_hessian_diag(features)
            return np.append(k_diag,k_hes_diag)
        return k_diag
        
    def get_gradients(self,features,hp,KXX,dis_m=None,correction=True,**kwargs):
        """Get the gradients of the kernel matrix in respect to the hyperparameters.
            Parameters:
                features : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array
                    The kernel matrix of training data.
                dis_m : (D,N,M) array (optional)
                    Already calculated distance matrix.
                correction : bool
                    Whether the noise correction is used.
        """
        hp_deriv={}
        if 'length' in hp:
            nd1=len(features)
            dim=int(len(KXX)/nd1)-1
            Kd=KXX.copy()
            if dis_m is None:
                dis_m=self.distances(features)
            D=self.distances.get_absolute_distances(dis_m)*np.exp(-2*self.hp['length']).item(0)
            Kd[:nd1,:nd1]=KXX[:nd1,:nd1]*D
            for d1 in range(1,dim+1):
                Kd[:nd1,nd1*d1:nd1*(d1+1)]=KXX[:nd1,nd1*d1:nd1*(d1+1)]*(D-2)
                Kd[nd1*d1:nd1*(d1+1),:nd1]=Kd[:nd1,nd1*d1:nd1*(d1+1)].copy().T
                for d2 in range(d1,dim+1):
                    Kdd1=self.get_hessian_K1(features,features,dis_m,KXX[:nd1,:nd1],d1,d2)
                    Kd[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]=KXX[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)]*(D-2)-(2*Kdd1)
                    if d1!=d2:
                        Kd[nd1*d2:nd1*(d2+1),nd1*d1:nd1*(d1+1)]=Kd[nd1*d1:nd1*(d1+1),nd1*d2:nd1*(d2+1)].copy().T
            if correction:
                Kd[range(nd1*dim),range(nd1*dim)]+=(-2*nd1*dim*np.exp(-2*self.hp['length']).item(0))*(1/(1/(4e-14)-(nd1*dim)))
            hp_deriv['length']=Kd
        return hp_deriv
    
    def get_dimension(self,features):
        " Get the dimension of the length-scale hyperparameter "
        return 1
    
    def __repr__(self):
        return 'SE(hp={})'.format(self.hp)
    

