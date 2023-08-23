import numpy as np
from scipy.spatial.distance import pdist,cdist

class Kernel:
    def __init__(self,use_derivatives=False,use_fingerprint=False,hp={}):
        """The Kernel class with hyperparameters.
            Parameters:
                use_derivatives: bool
                    Whether to use the derivatives of the targets.
                use_fingerprint: bool
                    Whether fingerprint objects is given or arrays.
                hp: dict
                    A dictionary of hyperparameters.
        """
        self.use_derivatives=use_derivatives
        self.use_fingerprint=use_fingerprint
        self.hp={'length':np.array([-0.7])}
        self.set_hyperparams(hp)

    def __call__(self,features,features2=None,get_derivatives=True,**kwargs):
        """Make the kernel matrix.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points and D dimensions. 
                    If it is not given a squared kernel from features is generated.
                get_derivatives: bool
                    Whether to predict derivatives of target.
        """
        if features2 is None:
            return self.get_KXX(features,**kwargs)
        return self.get_KQX(features,features2=features2,get_derivatives=get_derivatives,**kwargs)
    
    def get_KXX(self,features,**kwargs):
        """Make the symmetric kernel matrix.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
        """
        raise NotImplementedError()
    
    def get_KQX(self,features,features2,get_derivatives=True,**kwargs):
        """Make the kernel matrix.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
                features2 : (M,D) array or (M) list of fingerprint objects
                    Features with M data points and D dimensions. 
                    If it is not given a squared kernel from features is generated.
                get_derivatives: bool
                    Whether to predict derivatives of target.
        """
        raise NotImplementedError()
    
    def get_arrays(self,features,features2=None):
        " Get the feature matrix from the fingerprint "
        X=np.array([feature.get_vector() for feature in features])
        if features2 is None:
            return X
        Q=np.array([feature.get_vector() for feature in features2])
        return X,Q
    
    def get_symmetric_absolute_distances(self,features,metric='sqeuclidean'):
        " Calculate the symmetric absolute distance matrix in (scaled) feature space. "
        return pdist(features,metric=metric)

    def get_absolute_distances(self,features,features2,metric='sqeuclidean'):
        " Calculate the absolute distance matrix in (scaled) feature space. "
        return cdist(features,features2,metric=metric)
    
    def get_feature_dimension(self,features):
        " Get the dimension of the features "
        if self.use_fingerprint:
            return len(features[0].get_vector())
        return len(features[0])
    
    def get_fp_deriv(self,features,dim=None):
        " Get the derivatives of all the fingerprints. "
        if dim is None:
            return np.array([fp.get_derivatives() for fp in features]).transpose((2,0,1))
        return np.array([fp.get_derivatives(dim) for fp in features])

    def get_derivative_dimension(self,features):
        " Get the dimension of the features "
        if self.use_fingerprint:
            return int(features[0].get_derivative_dimension())
        return len(features[0])

    def diag(self,features,get_derivatives=True,**kwargs):
        """Get the diagonal kernel vector.
            Parameters:
                features : (N,D) array or (N) list of fingerprint objects
                    Features with N data points.
        """
        raise NotImplementedError()

    def get_gradients(self,features,hp,KXX,correction=True,**kwargs):
        """Get the gradients of the kernel matrix in respect to the hyperparameters.
            Parameters:
                features : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list of the hyperparameters that are optimized.
                KXX : (N,N) array
                    The kernel matrix of training data.
                correction : bool
                    Whether the noise correction is used.
        """
        raise NotImplementedError()
    
    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the Kernel.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
        """
        if 'length' in new_params:
            self.hp['length']=np.array(new_params['length'],dtype=float).reshape(-1)
        return self
    
    def get_hyperparams(self):
        " Get the hyperparameters for the kernel. "
        return {'length':self.hp['length'].copy()}
    
    def get_hp_dimension(self,features=None,**kwargs):
        " Get the dimension of the length-scale hyperparameter "
        return int(1)
    
    def copy(self):
        " Copy the kernel class object. "
        return self.__class__(use_derivatives=self.use_derivatives,use_fingerprint=self.use_fingerprint,hp=self.hp)
    
    def __repr__(self):
        return 'Kernel(use_derivatives={}, use_fingerprint={}, hp={})'.format(self.use_derivatives,self.use_fingerprint,self.hp)

