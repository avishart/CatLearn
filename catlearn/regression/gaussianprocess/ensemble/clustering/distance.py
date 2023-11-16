import numpy as np
from scipy.spatial.distance import cdist
from .clustering import Clustering

class DistanceClustering(Clustering):
    def __init__(self,metric='euclidean',centroids=np.array([]),**kwargs):
        """
        Clustering class object for data sets.
        Use distances to pre-defined centroids for clustering
        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
            centroids : (K,D) array
                An array with the centroids of the K clusters. 
                The centroids must have the same dimensions as the features.
        """
        # Set the arguments
        self.update_arguments(centroids=centroids,metric=metric,**kwargs)
    
    def cluster_fit_data(self,X,**kwargs):
        indicies=np.array(range(len(X)))
        i_min=np.argmin(self.calculate_distances(X,self.centroids),axis=1)
        return [indicies[i_min==ki] for ki in range(self.n_clusters)]
    
    def update_arguments(self,metric=None,centroids=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
            centroids : (K,D) array
                An array with the centroids of the K clusters. 
                The centroids must have the same dimensions as the features.
        Returns:
            self: The updated object itself.
        """
        if centroids is not None:
            self.set_centroids(centroids)
        if metric is not None:
            self.metric=metric
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(metric=self.metric,centroids=self.centroids)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    