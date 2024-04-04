import numpy as np
from scipy.spatial.distance import cdist
from .clustering import Clustering

class K_means(Clustering):
    def __init__(self,metric='euclidean',n_clusters=4,maxiter=100,tol=1e-4,**kwargs):
        """
        Clustering class object for data sets.
        The K-means++ algorithm for clustering.

        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
            n_clusters : int
                The number of used clusters.
            maxiter : int
                The maximum number of iterations used to fit the clusters.
            tol : float
                The tolerance before the cluster fit is converged.
        """
        # Set default descriptors
        self.centroids=np.array([])
        self.n_clusters=1
        # Set the arguments
        super().__init__(metric=metric,
                         n_clusters=n_clusters,
                         maxiter=maxiter,
                         tol=tol,
                         **kwargs)
    
    def cluster_fit_data(self,X,**kwargs):
        # If only one cluster is used give the full data 
        if self.n_clusters==1:
            self.centroids=np.array([np.mean(X,axis=0)])
            return [np.arange(len(X))]
        # Initiate the centroids
        centroids=self.initiate_centroids(X)
        # Optimize position of the centroids
        self.centroids=self.optimize_centroids(X,centroids)
        # Return the cluster indicies
        return self.cluster(X)
    
    def cluster(self,X,**kwargs):
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
    
    def update_arguments(self,metric=None,n_clusters=None,maxiter=None,tol=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
            n_clusters : int
                The number of used clusters.
            maxiter : int
                The maximum number of iterations used to fit the clusters.
            tol : float
                The tolerance before the cluster fit is converged.
                
        Returns:
            self: The updated object itself.
        """
        if metric is not None:
            self.metric=metric
        if n_clusters is not None:
            self.n_clusters=int(n_clusters)
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if tol is not None:
            self.tol=tol
        return self
    
    def calculate_distances(self,Q,X,**kwargs):
        " Calculate the distances. "
        return cdist(Q,X,metric=self.metric)
    
    def initiate_centroids(self,X,**kwargs):
        " Initial the centroids from the K-mean++ method. "
        # Get the first centroid randomly 
        centroids=np.array(X[np.random.choice(len(X),size=1)])
        for ki in range(1,self.n_clusters):
            # Calculate the maximum nearest neighbor
            i_max=np.argmax(np.min(self.calculate_distances(X,centroids),axis=1))
            centroids=np.append(centroids,[X[i_max]],axis=0)
        return centroids
    
    def optimize_centroids(self,X,centroids,**kwargs):
        " Optimize the positions of the centroids. "
        for i in range(1,self.maxiter+1):
            # Store the old centroids
            centroids_old=centroids.copy()
            # Calculate which centroids that are closest
            i_min=np.argmin(self.calculate_distances(X,centroids),axis=1)
            centroids=np.array([np.mean(X[i_min==ki],axis=0) for ki in range(self.n_clusters)])
            # Check if it is converged
            if np.linalg.norm(centroids-centroids_old)<=self.tol:
                break
        return centroids
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(metric=self.metric,
                        n_clusters=self.n_clusters,
                        maxiter=self.maxiter,
                        tol=self.tol)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict(centroids=self.centroids)
        return arg_kwargs,constant_kwargs,object_kwargs
