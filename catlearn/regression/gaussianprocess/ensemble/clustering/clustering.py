
import numpy as np

class Clustering:
    def __init__(self,**kwargs):
        """
        Clustering class object for data sets.
        """
        # Set default descriptors
        self.n_clusters=1
        # Set the arguments
        self.update_arguments(**kwargs)
        
    def fit(self,X,**kwargs):
        """
        Fit the clustering algorithm.

        Parameters:
            X : (N,D) array
                Training features with N data points.

        Returns:
            self: The fitted object itself.
        """
        # Cluster the training data
        self.cluster_fit_data(X,**kwargs)
        return self
    
    def cluster_fit_data(self,X,**kwargs):
        """
        Fit the clustering algorithm and return the clustered data.

        Parameters:
            X : (N,D) array
                Training features with N data points.

        Returns:
            list: A list of indicies to the training data for each cluster.
        """
        raise NotImplementedError()
        
    def cluster(self,X,**kwargs):
        """
        Cluster the given data if it is fitted.

        Parameters:
            X : (M,D) array
                Features with M data points.

        Returns:
            list: A list of indicies to the data for each cluster.
        """
        raise NotImplementedError()
    
    def update_arguments(self,metric=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Returns:
            self: The updated object itself.
        """
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict()
        # Get the constants made within the class
        constant_kwargs=dict(n_clusters=self.n_clusters)
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
    