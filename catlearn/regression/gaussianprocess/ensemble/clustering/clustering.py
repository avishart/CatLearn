from scipy.spatial.distance import cdist
import numpy as np

class Clustering:
    def __init__(self,metric='euclidean',**kwargs):
        " Clustering class object for data sets. "
        self.metric=metric
        
    def fit(self,X,**kwargs):
        " Fit the clustering algorithm. "
        raise NotImplementedError()
        
    def cluster(self,X,**kwargs):
        " Cluster the data if it is fitted. "
        indicies=np.arange(len(X))
        i_min=np.argmin(self.calculate_distances(X,self.centroids),axis=1)
        return [indicies[i_min==ki] for ki in range(self.k)]
        
    def cluster_fit_data(self,X,**kwargs):
        " Cluster the data used for fitting the algorithm. "
        raise NotImplementedError()
        
    def calculate_distances(self,Q,X,**kwargs):
        " Calculate the distances. "
        return cdist(Q,X,metric=self.metric)
        
    def copy(self):
        " Copy the cluster object. "
        return self.__class__(metric=self.metric)
        
    def __repr__(self):
        return "Clustering(metric={})".format(self.metric)
    