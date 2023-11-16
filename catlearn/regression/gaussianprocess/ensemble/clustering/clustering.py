from scipy.spatial.distance import cdist
import numpy as np

class Clustering:
    def __init__(self,metric='euclidean',**kwargs):
        """
        Clustering class object for data sets.
        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
        """
        # Set default descriptors
        self.centroids=np.array([])
        self.n_clusters=0
        # Set the arguments
        self.update_arguments(metric=metric,**kwargs)
        
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
        cluster_indicies=self.cluster_fit_data(X,**kwargs)
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
        indicies=np.arange(len(X))
        i_min=np.argmin(self.calculate_distances(X,self.centroids),axis=1)
        return [indicies[i_min==ki] for ki in range(self.n_clusters)]

    def set_centroids(self,centroids,**kwargs):
        """
        Set user defined centroids. 
        Parameters:
            centroids : (K,D) array
                An array with the centroids of the K clusters. 
                The centroids must have the same dimensions as the features.
        Returns:
            self: The updated object itself.
        """
        self.centroids=centroids.copy()
        self.n_clusters=len(self.centroids)
        return self
    
    def update_arguments(self,metric=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
        Returns:
            self: The updated object itself.
        """
        if metric is not None:
            self.metric=metric
        return self
        
    def calculate_distances(self,Q,X,**kwargs):
        " Calculate the distances. "
        return cdist(Q,X,metric=self.metric)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(metric=self.metric)
        # Get the constants made within the class
        constant_kwargs=dict(n_clusters=self.n_clusters)
        # Get the objects made within the class
        object_kwargs=dict(centroids=self.centroids)
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
    