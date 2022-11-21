import numpy as np
from scipy.spatial.distance import pdist,cdist

class Distance_matrix:
    def __init__(self,use_fingerprint=False):
        """ The absolute distance matrix from a matrix """
        self.use_derivatives=False
        self.use_fingerprint=use_fingerprint
        
    def __call__(self,features,features2=None,scale=False,**kwargs):
        " Get the squared absolute distance matrix "
        if self.use_fingerprint:
            features,features2=self.get_arrays(features,features2)
        if scale:
            features,features2=self.scales(features,features2,scale)
        if features2 is None:
            return pdist(features,metric='sqeuclidean')
        return cdist(features,features2,metric='sqeuclidean')
    
    def scales(self,X,Q=None,scale=1.0):
        " Scale the features with a length-scale"
        if Q is not None:
            Q=Q.copy()*scale
        X=X.copy()*scale
        return X,Q
    
    def get_arrays(self,features,features2=None):
        " Get the feature matrix from the fingerprint "
        X=np.array([feature.get_vector() for feature in features])
        if features2 is None:
            return X,None
        Q=np.array([feature.get_vector() for feature in features2])
        return X,Q
    
    def get_dimension(self,features):
        " Get the dimension of the features "
        if self.use_fingerprint:
            return len(features[0].get_vector())
        return len(features[0])
    
    def get_diag(self,features):
        " Get the diagonal elements of the distance matrix. "
        return np.array([0.0]*len(features))

    
class Distance_matrix_per_dimension(Distance_matrix):
    def __init__(self,use_fingerprint=False):
        """ The absolute distance matrix from a matrix """
        self.use_derivatives=True
        self.use_fingerprint=use_fingerprint
        
    def __call__(self,features,features2=None,scale=False,**kwargs):
        " Get the distance matrix for each dimension "
        if self.use_fingerprint:
            features,features2=self.get_arrays(features,features2)
        dim=len(features[0])
        if scale:
            features,features2=self.scales(features,features2,scale)
        if features2 is None:
            return np.array([features[:,d:d+1]-features[:,d] for d in range(dim)])
        return np.array([features[:,d:d+1]-features2[:,d] for d in range(dim)])
    
    def get_absolute_distances(self,distances):
        " Get the absolute distance matrix with the right metric "
        return np.sum(distances**2,axis=0)
    
    def get_derivative(self,features,distances,d1,axis=0):
        " Get the derivative of the distance matrix wrt the features/fingerprint "
        if self.use_fingerprint:
            fp_deriv=np.array([fp.get_derivatives(d1) for fp in features])
            if axis==0:
                return 2.0*np.sum([distances[d]*fp_deriv[:,d:d+1] for d in range(len(distances))],axis=0)
            return (-2.0)*np.sum([distances[d]*fp_deriv[:,d] for d in range(len(distances))],axis=0)
        if axis==0:
            return 2.0*distances[d1]
        return (-2.0)*distances[d1]
    
    def get_hessian(self,features,features2,distances,d1,d2):
        " Get the hessian of the distance matrix wrt the features/fingerprint "
        if self.use_fingerprint:
            fp_deriv=np.array([fp.get_derivatives(d1) for fp in features])
            fp2_deriv=np.array([fp.get_derivatives(d2) for fp in features2])
            return (-2.0)*np.sum([fp_deriv[:,d:d+1]*fp2_deriv[:,d] for d in range(len(distances))],axis=0)
        return -2.0 if d1==d2 else 0
    
    def get_derivative_dimension(self,features):
        " Get the dimension of the features "
        if self.use_fingerprint:
            return features[0].get_derivative_dimension()
        return len(features[0])
    
    def get_diag(self,features):
        " Get the diagonal elements of the distance matrix. "
        return np.array([0.0]*len(features))
    
    def get_derivative_diag(self,features):
        " Get the diagonal elements of the distance matrix. "
        n_data=len(features)
        if self.use_fingerprint:
            dim=features[0].get_derivative_dimension()
            return np.array([0.0]*dim*n_data)
        return np.array([0.0]*len(features[0])*n_data)
    
    def get_hessian_diag(self,features):
        " Get the diagonal elements of the distance matrix. "
        n_data=len(features)
        if self.use_fingerprint:
            dim=self.get_derivative_dimension(features)
            return -2*np.array([[np.sum(feature.get_derivatives(d)**2) for feature in features] for d in range(dim)]).reshape(-1)
        return np.array([-2.0]*len(features[0])*n_data)
    


