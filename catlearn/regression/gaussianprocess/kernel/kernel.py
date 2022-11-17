import numpy as np
import copy
from .distances import Distance_matrix,Distance_matrix_per_dimension

class Kernel:
    def __init__(self,distances=None,use_fingerprint=False,hp={'length':np.array([0.0])}):
        """The Kernel class with hyperparameters.
            Parameters:
                distances : Distance class
                    A distance matrix object that calculates the distances.
                use_fingerprint : bool
                    Whether fingerprint objects is given or arrays.
                hp : dict
                    A dictionary of hyperparameters.
        """
        self.use_derivatives=False
        if distances is None:
            distances=Distance_matrix(use_fingerprint=use_fingerprint)
        self.distances=copy.deepcopy(distances)
        self.use_fingerprint=use_fingerprint
        self.check_attributes()
        self.hp=hp.copy()
        self.set_hyperparams(hp)
    
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
        raise NotImplementedError()

    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
        return self.hp
    
    def __call__(self,features,features2=None,get_derivatives=True,dis_m=None,**kwargs):
        """Make the kernel matrix.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points and D dimensions. 
                    If it is not given a squared kernel from features is generated.
                get_derivatives : bool
                    Can only be False.
                dis_m : (N,M) or (N*(N-1)/2) array (optional)
                    Already calculated distance matrix.
        """
        return self.get_K(features,features2,dis_m,**kwargs)

    def diag(self,features,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
        """
        raise NotImplementedError()

    def get_gradients(self,features,hp,KXX,dis_m=None,**kwargs):
        """Get the gradients of the kernel matrix in respect to the hyperparameters.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array
                    The kernel matrix of training data .
                dis_m : (N,N) array (optional)
                    Already calculated distance matrix.
        """
        raise NotImplementedError()

    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.use_fingerprint!=self.distances.use_fingerprint:
            raise Exception('Kernel and Distances do not agree whether to use fingerprints!')
        if self.use_derivatives!=self.distances.use_derivatives:
            raise Exception('Kernel and Distances do not agree whether to use derivatives!')
        return
    
    def get_dimension(self,features):
        " Get the dimension of the length-scale hyperparameter "
        return 1

    def copy(self):
        " Deepcopy the object "
        return copy.deepcopy(self)
    
    def __repr__(self):
        return 'Kernel(hp={})'.format(self.hp)


class Kernel_Derivative:
    def __init__(self,distances=None,use_fingerprint=False,hp={'length':np.array([0.0])}):
        """The Kernel class with hyperparameters.
            Parameters:
                distances : Distance class
                    A distance matrix object that calculates the distances.
                use_fingerprint : bool
                    Whether fingerprint objects is given or arrays.
                hp : dict
                    A dictionary of hyperparameters.
        """
        self.use_derivatives=True
        if distances is None:
            distances=Distance_matrix_per_dimension(use_fingerprint=use_fingerprint)
        self.distances=copy.deepcopy(distances)
        self.use_fingerprint=use_fingerprint
        self.check_attributes()
        self.hp=hp.copy()
        self.set_hyperparams(hp)
        
    def get_K(self,dis_m,**kwargs):
        """Make the kernel matrix without derivatives.
            Parameters:
                dis_m : (N,M) or (N*(N-1)/2) array
                    Already calculated distance matrix.
        """
        raise NotImplementedError()
        
    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        self.hp.update(new_params)
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
        raise NotImplementedError()
    
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
        raise NotImplementedError()
    
    def get_KXX(self,features,dis_m=None,**kwargs):
        """ Get the symmetric kernel matrix. 
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                dis_m : (D,N,M) array (optional)
                    Already calculated distance matrix.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
        
    def __call__(self,features,features2=None,get_derivatives=False,dis_m=None,**kwargs):
        """Make the kernel matrix.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points and D dimensions. 
                    If it is not given a squared kernel from features is generated.
                get_derivatives : bool
                    Whether to get the derivatives of the prediction part.
                dis_m : (D,N,M) array (optional)
                    Already calculated distance matrix.
        """
        if features2 is None:
            return self.get_KXX(features,dis_m=dis_m,**kwargs)
        return self.get_KQX(features,features2,get_derivatives=get_derivatives,dis_m=dis_m,**kwargs)

    def diag(self,features,get_derivatives=True):
        """Get the diagonal kernel vector.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                get_derivatives : bool
                    Whether to get the derivatives of the prediction part.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.use_fingerprint!=self.distances.use_fingerprint:
            raise Exception('Kernel and Distances do not agree whether to use fingerprints!')
        if self.use_derivatives!=self.distances.use_derivatives:
            raise Exception('Kernel and Distances do not agree whether to use derivatives!')
        return
    
    def get_dimension(self,features):
        " Get the dimension of the length-scale hyperparameter "
        return 1

    def copy(self):
        " Deepcopy the object "
        return copy.deepcopy(self)
    
    def __repr__(self):
        return 'Kernel(hp={})'.format(self.hp)
