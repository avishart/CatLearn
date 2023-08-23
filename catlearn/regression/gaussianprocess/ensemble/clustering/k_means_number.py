import numpy as np
from .clustering import Clustering

class K_means_number(Clustering):
    def __init__(self,data_number=40,maxiter=100,tol=1e-4,metric='euclidean',**kwargs):
        " The K-means++ algorithm for clustering, but where the number of clusters are updated from a fixed number data point in each cluster. "
        super().__init__(metric=metric,**kwargs)
        self.data_number=int(data_number)
        self.maxiter=int(maxiter)
        self.tol=tol
        
    def fit(self,X,**kwargs):
        " Fit the clustering algorithm. "
        n_data=len(X)
        self.k=int(n_data//self.data_number)
        if n_data-(self.k*self.data_number):
            self.k=self.k+1
        if self.k==1:
            self.centroids=np.array([np.mean(X,axis=0)])
            return self
        centroids=self.initiate_centroids(X)
        self.centroids,cluster_indicies=self.optimize_centroids(X,centroids)
        return self
    
    def cluster_fit_data(self,X,**kwargs):
        " Cluster the data used for fitting the algorithm. "
        n_data=len(X)
        self.k=int(n_data//self.data_number)
        if n_data-(self.k*self.data_number):
            self.k=self.k+1
        if self.k==1:
            self.centroids=np.array([np.mean(X,axis=0)])
            return [list(range(n_data))]
        centroids=self.initiate_centroids(X)
        self.centroids,cluster_indicies=self.optimize_centroids(X,centroids)
        return cluster_indicies
    
    def initiate_centroids(self,X,**kwargs):
        " Initial centroids from K-mean++ method. "
        # Get the first centroid randomly 
        centroids=np.array(X[np.random.choice(len(X),size=1)])
        for ki in range(1,self.k):
            # Calculate the maximum nearest neighbor
            i_max=np.argmax(np.min(self.calculate_distances(X,centroids),axis=1))
            centroids=np.append(centroids,[X[i_max]],axis=0)
        return centroids
    
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
        """ Get the indicies for each of the clusters.
            The number of data points in each cluster is counted and restricted 
            between the minimum and maximum number of allowed cluster sizes. """
        # Make a list cluster indicies
        klist=np.arange(self.k).reshape(-1,1)
        # Find the cluster that each point is closest to
        k_indicies=np.argmin(distance_matrix,axis=1)
        indicies_ki_bool=(klist==k_indicies)
        # Sort the indicies as function of the distances to the centroids
        d_indicies=np.argsort(distance_matrix,axis=0)
        indicies_sorted=indicies[d_indicies.T]
        indicies_ki_bool=indicies_ki_bool[klist,indicies_sorted]
        # Prioritize the points that is part of each cluster
        cluster_indicies=[np.append(indicies_sorted[ki,indicies_ki_bool[ki]],indicies_sorted[ki,~indicies_ki_bool[ki]])[:self.data_number] for ki in range(self.k)]
        return cluster_indicies
    
    def set_centroids(self,centroids):
        " Set the centroids. "
        self.centroids=centroids.copy()
        return self
    
    def copy(self):
        " Copy the cluster object. "
        clone=self.__class__(data_number=self.data_number,maxiter=self.maxiter,tol=self.tol,metric=self.metric)
        if 'k' in self.__dict__.keys():
            clone.k=self.k
        if 'centroids' in self.__dict__.keys():
            clone.centroids=self.centroids.copy()
        return clone
        
    def __repr__(self):
        return "K_means_number(data_number={},maxiter={},tol={},metric={})".format(self.data_number,self.maxiter,self.tol,self.metric)
        