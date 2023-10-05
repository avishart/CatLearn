import numpy as np
from scipy.spatial.distance import pdist,cdist

class Kernel:
    def __init__(self,use_derivatives=False,use_fingerprint=False,hp={},**kwargs):
        """
        The Kernel class with hyperparameters.
        Parameters:
            use_derivatives: bool
                Whether to use the derivatives of the targets.
            use_fingerprint: bool
                Whether fingerprint objects is given or arrays.
            hp: dict
                A dictionary of the hyperparameters in the log-space.
                The hyperparameters should be given as flatten arrays, 
                like hp=dict(length=np.array([-0.7])).
        """
        # Set the default hyperparameters
        self.hp=dict(length=np.array([-0.7]))
        # Set all the arguments
        self.update_arguments(use_derivatives=use_derivatives,
                              use_fingerprint=use_fingerprint,
                              hp=hp,
                              **kwargs)

    def __call__(self,features,features2=None,get_derivatives=True,**kwargs):
        """
        Make the kernel matrix.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Features with N data points.
            features2 : (M,D) array or (M) list of fingerprint objects
                Features with M data points and D dimensions. 
                If it is not given a squared kernel from features is generated.
            get_derivatives: bool
                Whether to predict derivatives of target.
        Returns:
            KXX : array
                The symmetric kernel matrix if features2=None.
                The number of rows in the array is N, or N*(D+1) if get_derivatives=True.
                The number of columns in the array is N, or N*(D+1) if use_derivatives=True.
            or
            KQX : array
                The kernel matrix if features2 is not None.
                The number of rows in the array is N, or N*(D+1) if get_derivatives=True.
                The number of columns in the array is M, or M*(D+1) if use_derivatives=True.
        """
        if features2 is None:
            return self.get_KXX(features,**kwargs)
        return self.get_KQX(features,features2=features2,get_derivatives=get_derivatives,**kwargs)
    
    def diag(self,features,get_derivatives=True,**kwargs):
        """
        Get the diagonal kernel vector.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Features with N data points.
            get_derivatives: bool
                Whether to predict derivatives of target.
        Returns:
            (N) or (N*D+1) array: The diagonal elements of the symmetric kernel matrix.
        """
        raise NotImplementedError()

    def get_gradients(self,features,hp,KXX,correction=True,**kwargs):
        """
        Get the gradients of the kernel matrix in respect to the hyperparameters.
        Parameters:
            features : (N,D) array
                Features with N data points and D dimensions.
            hp : list
                A list of the string names of the hyperparameters that are optimized.
            KXX : (N,N) array
                The kernel matrix of training data.
            correction : bool
                Whether the noise correction is used.
        Returns:
            dict: A dictionary with gradient of the symmetric kernel matrix wrt. the hyperparameter.
        """
        raise NotImplementedError()
    
    def set_hyperparams(self,new_params,**kwargs):
        """
        Set or update the hyperparameters for the Kernel.
        Parameters:
            new_params: dictionary
                A dictionary of hyperparameters in the log-space that are added or updated.
        Returns:
            self: The updated object itself.
        """
        if 'length' in new_params:
            self.hp['length']=np.array(new_params['length'],dtype=float).reshape(-1)
        return self
    
    def get_hyperparams(self,**kwargs):
        """
        Get the hyperparameters for the kernel.
        Returns:
            dict: The hyperparameters in the log-space from the kernel class.  
        """
        return {'length':self.hp['length'].copy()}
    
    def get_hp_dimension(self,features=None,**kwargs):
        """
        Get the dimension of the length-scale hyperparameter.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects or None
                Features with N data points.
        Returns:
            int: The dimensions of the length-scale hyperparameter.  
        """
        return int(1)
    
    def update_arguments(self,use_derivatives=None,use_fingerprint=None,hp=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            use_derivatives: bool
                Whether to use the derivatives of the targets.
            use_fingerprint: bool
                Whether fingerprint objects is given or arrays.
            hp: dict
                A dictionary of the hyperparameters in the log-space.
                The hyperparameters should be given as flatten arrays, 
                like hp=dict(length=np.array([-0.7])).
        Returns:
            self: The updated object itself.
        """
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        if use_fingerprint is not None:
            self.use_fingerprint=use_fingerprint
        if hp is not None:
            self.set_hyperparams(hp)
        return self
    
    def get_KXX(self,features,**kwargs):
        """
        Make the symmetric kernel matrix.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Features with N data points.
        Returns:
            KXX : array
                The symmetric kernel matrix if features2=None.
                The number of rows in the array is N, or N*(D+1) if get_derivatives=True.
                The number of columns in the array is N, or N*(D+1) if use_derivatives=True.
        """
        raise NotImplementedError()
    
    def get_KQX(self,features,features2,get_derivatives=True,**kwargs):
        """
        Make the kernel matrix.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Features with N data points.
            features2 : (M,D) array or (M) list of fingerprint objects
                Features with M data points and D dimensions. 
                If it is not given a squared kernel from features is generated.
            get_derivatives: bool
                Whether to predict derivatives of target.
        Returns:
            KQX : array
                The kernel matrix if features2 is not None.
                The number of rows in the array is N, or N*(D+1) if get_derivatives=True.
                The number of columns in the array is M, or M*(D+1) if use_derivatives=True.
        """
        raise NotImplementedError()
    
    def get_arrays(self,features,features2=None,**kwargs):
        " Get the feature matrix from the fingerprint. "
        X=np.array([feature.get_vector() for feature in features])
        if features2 is None:
            return X
        Q=np.array([feature.get_vector() for feature in features2])
        return X,Q
    
    def get_symmetric_absolute_distances(self,features,metric='sqeuclidean',**kwargs):
        " Calculate the symmetric absolute distance matrix in (scaled) feature space. "
        return pdist(features,metric=metric)

    def get_absolute_distances(self,features,features2,metric='sqeuclidean',**kwargs):
        " Calculate the absolute distance matrix in (scaled) feature space. "
        return cdist(features,features2,metric=metric)
    
    def get_feature_dimension(self,features,**kwargs):
        " Get the dimension of the features. "
        if self.use_fingerprint:
            return len(features[0].get_vector())
        return len(features[0])
    
    def get_fp_deriv(self,features,dim=None,**kwargs):
        " Get the derivatives of all the fingerprints. "
        if dim is None:
            return np.array([fp.get_derivatives() for fp in features]).transpose((2,0,1))
        return np.array([fp.get_derivatives(dim) for fp in features])

    def get_derivative_dimension(self,features,**kwargs):
        " Get the dimension of the features. "
        if self.use_fingerprint:
            return int(features[0].get_derivative_dimension())
        return len(features[0])

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(use_derivatives=self.use_derivatives,
                        use_fingerprint=self.use_fingerprint,
                        hp=self.hp)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    
    def copy(self):
        " Copy the object. "
        # Get all arguments
        arg_kwargs,constant_kwargs,object_kwargs=self.get_arguments()
        # Make a clone
        clone=self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key,value in constant_kwargs.items():
                clone.__dict__[key]=value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key,value in object_kwargs.items():
                clone.__dict__[key]=value.copy()
        return clone
    
    def __repr__(self):
        arg_kwargs=self.get_arguments()[0]
        str_kwargs=",".join([f"{key}={value}" for key,value in arg_kwargs.items()])
        return "{}({})".format(self.__class__.__name__,str_kwargs)
    