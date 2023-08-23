import numpy as np
from .clustering import Clustering

class K_means(Clustering):
    def __init__(self,k=4,maxiter=100,tol=1e-4,metric='euclidean',**kwargs):
        " The K-means++ algorithm for clustering. "
        super().__init__(metric=metric,**kwargs)
        self.k=int(k)
        self.maxiter=int(maxiter)
        self.tol=tol
        
    def fit(self,X,**kwargs):
        " Fit the clustering algorithm. "
        if self.k==1:
            self.centroids=np.array([np.mean(X,axis=0)])
            return self
        centroids=self.initiate_centroids(X)
        self.centroids=self.optimize_centroids(X,centroids)
        return self
    
    def cluster_fit_data(self,X,**kwargs):
        " Cluster the data used for fitting the algorithm. "
        if self.k==1:
            self.centroids=np.array([np.mean(X,axis=0)])
            return [list(range(len(X)))]
        self.fit(X)
        return self.cluster(X)
    
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
        for i in range(1,self.maxiter+1):
            # Store the old centroids
            centroids_old=centroids.copy()
            # Calculate which centroids that are closest
            i_min=np.argmin(self.calculate_distances(X,centroids),axis=1)
            centroids=np.array([np.mean(X[i_min==ki],axis=0) for ki in range(self.k)])
            # Check if it is converged
            if np.linalg.norm(centroids-centroids_old)<=self.tol:
                break
        return centroids
    
    def set_centroids(self,centroids):
        " Set the centroids. "
        self.centroids=centroids.copy()
        return self
    
    def copy(self):
        " Copy the cluster object. "
        clone=self.__class__(k=self.k,maxiter=self.maxiter,tol=self.tol,metric=self.metric)
        if 'centroids' in self.__dict__.keys():
            clone.centroids=self.centroids.copy()
        return clone
        
    def __repr__(self):
        return "K_means(k={},maxiter={},tol={},metric={})".format(self.k,self.maxiter,self.tol,self.metric)