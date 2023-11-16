import numpy as np
from .k_means import K_means

class K_means_number(K_means):
    def __init__(self,metric='euclidean',data_number=25,maxiter=100,tol=1e-4,**kwargs):
        """
        Clustering class object for data sets.
        The K-means++ algorithm for clustering, but where the number of clusters are updated from a fixed number data point in each cluster.
        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
            data_number : int
                The number of data point in each cluster.
            maxiter : int
                The maximum number of iterations used to fit the clusters.
            tol : float
                The tolerance before the cluster fit is converged.
        """
        super().__init__(metric=metric,
                         data_number=data_number,
                         maxiter=maxiter,
                         tol=tol,
                         **kwargs)
        
    def cluster_fit_data(self,X,**kwargs):
        # Calculate the number of clusters
        n_data=len(X)
        self.n_clusters=int(n_data//self.data_number)
        if n_data-(self.n_clusters*self.data_number):
            self.n_clusters=self.n_clusters+1
        # If only one cluster is used give the full data 
        if self.n_clusters==1:
            self.centroids=np.array([np.mean(X,axis=0)])
            return [np.arange(n_data)]
        # Initiate the centroids
        centroids=self.initiate_centroids(X)
        # Optimize position of the centroids
        self.centroids,cluster_indicies=self.optimize_centroids(X,centroids)
        # Return the cluster indicies
        return cluster_indicies
    
    def update_arguments(self,metric=None,data_number=None,maxiter=None,tol=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            metric : str
                The metric used to calculate the distances of the data.
            data_number : int
                The number of data point in each cluster.
            maxiter : int
                The maximum number of iterations used to fit the clusters.
            tol : float
                The tolerance before the cluster fit is converged.
        Returns:
            self: The updated object itself.
        """
        if metric is not None:
            self.metric=metric
        if data_number is not None:
            self.data_number=int(data_number)
        if maxiter is not None:
            self.maxiter=int(maxiter)
        if tol is not None:
            self.tol=tol
        return self
    
    def optimize_centroids(self,X,centroids,**kwargs):
        " Optimize the positions of the centroids. "
        indicies=np.arange(len(X))
        for i in range(1,self.maxiter+1):
            # Store the old centroids
            centroids_old=centroids.copy()
            # Calculate which centroids that are closest
            distance_matrix=self.calculate_distances(X,centroids)
            cluster_indicies=self.count_clusters(X,indicies,distance_matrix)
            centroids=np.array([np.mean(X[indicies_ki],axis=0) for indicies_ki in cluster_indicies])
            # Check if it is converged
            if np.linalg.norm(centroids-centroids_old)<=self.tol:
                break
        return centroids,cluster_indicies
    
    def count_clusters(self,X,indicies,distance_matrix,**kwargs):
        """ 
        Get the indicies for each of the clusters.
        The number of data points in each cluster is counted and restricted 
        between the minimum and maximum number of allowed cluster sizes. 
        """
        # Make a list cluster indicies
        klist=np.arange(self.n_clusters).reshape(-1,1)
        # Find the cluster that each point is closest to
        k_indicies=np.argmin(distance_matrix,axis=1)
        indicies_ki_bool=(klist==k_indicies)
        # Sort the indicies as function of the distances to the centroids
        d_indicies=np.argsort(distance_matrix,axis=0)
        indicies_sorted=indicies[d_indicies.T]
        indicies_ki_bool=indicies_ki_bool[klist,indicies_sorted]
        # Prioritize the points that is part of each cluster
        cluster_indicies=[np.append(indicies_sorted[ki,indicies_ki_bool[ki]],indicies_sorted[ki,~indicies_ki_bool[ki]])[:self.data_number] for ki in range(self.n_clusters)]
        return cluster_indicies

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(metric=self.metric,
                        data_number=self.data_number,
                        maxiter=self.maxiter,
                        tol=self.tol)
        # Get the constants made within the class
        constant_kwargs=dict(n_clusters=self.n_clusters)
        # Get the objects made within the class
        object_kwargs=dict(centroids=self.centroids)
        return arg_kwargs,constant_kwargs,object_kwargs
