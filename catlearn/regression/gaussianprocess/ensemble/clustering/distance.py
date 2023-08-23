import numpy as np
from scipy.spatial.distance import cdist
from .clustering import Clustering

class DistanceClustering(Clustering):
    def __init__(self,centroids,metric='euclidean',**kwargs):
        " Use distances to pre-defined centroids for clustering. "
        super().__init__(metric=metric,**kwargs)
        self.centroids=centroids.copy()
        self.k=len(self.centroids)
        
    def fit(self,X,**kwargs):
        " Fit the clustering algorithm. "
        return self
    
    def cluster_fit_data(self,X,**kwargs):
        " Cluster the data used for fitting the algorithm. "
        indicies=np.array(range(len(X)))
        i_min=np.argmin(self.calculate_distances(X,self.centroids),axis=1)
        return [indicies[i_min==ki] for ki in range(self.k)]
    
    def set_centroids(self,centroids):
        " Set the centroids. "
        self.centroids=centroids.copy()
        return self
    
    def copy(self):
        " Copy the cluster object. "
        return self.__class__(centroids=self.centroids,metric=self.metric)
        
    def __repr__(self):
        return "DistanceClustering(centroids={},metric={})".format(self.centroids,self.metric)
