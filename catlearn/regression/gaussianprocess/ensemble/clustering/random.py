import numpy as np
from .clustering import Clustering

class RandomClustering(Clustering):
    def __init__(self,n_clusters=4,equal_size=True,seed=None,**kwargs):
        """
        Clustering class object for data sets.
        The K-means++ algorithm for clustering.

        Parameters:
            n_clusters : int
                The number of used clusters.
            equal_size : bool
                Whether the clusters are forced to have the same size.
            seed : int (optional)
                The random seed used to permute the indicies. 
                If seed=None or False or 0, a random seed is not used.
        """
        # Set a random seed
        self.seed=seed
        super().__init__(n_clusters=n_clusters,
                         equal_size=equal_size,
                         seed=seed,
                         **kwargs)
    
    def cluster_fit_data(self,X,**kwargs):
        # Make indicies
        n_data=len(X)
        indicies=np.arange(n_data)
        # If only one cluster is used give the full data 
        if self.n_clusters==1:
            return [indicies]
        # Randomly make clusters
        i_clusters=self.randomized_clusters(indicies,n_data)
        # Return the cluster indicies
        return i_clusters
    
    def cluster(self,X,**kwargs):
        return self.cluster_fit_data(X)
    
    def update_arguments(self,n_clusters=None,equal_size=None,seed=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters:
            n_clusters : int
                The number of used clusters.
            equal_size : bool
                Whether the clusters are forced to have the same size.
            seed : int (optional)
                The random seed used to permute the indicies. 
                If seed=None or False or 0, a random seed is not used.

        Returns:
            self: The updated object itself.
        """
        if n_clusters is not None:
            self.n_clusters=int(n_clusters)
        if equal_size is not None:
            self.equal_size=equal_size
        if seed is not None:
            self.seed=seed
        return self
    
    def randomized_clusters(self,indicies,n_data,**kwargs):
        " Randomized indicies used for each cluster. "
        # Permute the indicies
        i_perm=self.get_permutation(indicies)
        # Ensure equal sizes of clusters if chosen
        if self.equal_size:
            i_perm=self.ensure_equal_sizes(i_perm,n_data)
        i_clusters=np.array_split(i_perm,self.n_clusters)
        return i_clusters
    
    def get_permutation(self,indicies):
        " Permute the indicies "
        if self.seed:
            rng=np.random.default_rng(seed=self.seed)
            return rng.permutation(indicies)
        return np.random.permutation(indicies)
    
    def ensure_equal_sizes(self,i_perm,n_data,**kwargs):
        " Extend the permuted indicies so the clusters have equal sizes. "
        # Find the number of excess points left
        n_left=n_data%self.n_clusters
        # Find the number of points that should be added
        if n_left>0:
            n_missing=self.n_clusters-n_left
        else:
            n_missing=0
        # Extend the permuted indicies
        if n_missing>0:
            if n_missing>n_data:
                i_perm=np.append(i_perm,np.tile(i_perm,(n_missing//n_data)+1)[:n_missing])
            else:
                i_perm=np.append(i_perm,i_perm[:n_missing])
        return i_perm
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(n_clusters=self.n_clusters,
                        equal_size=self.equal_size,
                        seed=self.seed)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
