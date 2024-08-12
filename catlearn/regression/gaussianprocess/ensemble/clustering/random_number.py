import numpy as np
from .random import RandomClustering


class RandomClustering_number(RandomClustering):
    def __init__(self, data_number=25, seed=None, **kwargs):
        """
        Clustering class object for data sets.
        The K-means++ algorithm for clustering.

        Parameters:
            data_number : int
                The number of data point in each cluster.
            seed : int (optional)
                The random seed used to permute the indicies.
                If seed=None or False or 0, a random seed is not used.
        """
        super().__init__(data_number=data_number, seed=seed, **kwargs)

    def cluster_fit_data(self, X, **kwargs):
        # Make indicies
        n_data = len(X)
        indicies = np.arange(n_data)
        # Calculate the number of clusters
        self.n_clusters = int(n_data // self.data_number)
        if n_data - (self.n_clusters * self.data_number):
            self.n_clusters = self.n_clusters + 1
        # If only one cluster is used give the full data
        if self.n_clusters == 1:
            return [indicies]
        # Randomly make clusters
        i_clusters = self.randomized_clusters(indicies, n_data)
        # Return the cluster indicies
        return i_clusters

    def update_arguments(self, data_number=None, seed=None, **kwargs):
        """
        Update the class with its arguments. The existing arguments are used
        if they are not given.

        Parameters:
            data_number : int
                The number of data point in each cluster.
            seed : int (optional)
                The random seed used to permute the indicies.
                If seed=None or False or 0, a random seed is not used.

        Returns:
            self: The updated object itself.
        """
        if data_number is not None:
            self.data_number = int(data_number)
        if seed is not None:
            self.seed = seed
        return self

    def randomized_clusters(self, indicies, n_data, **kwargs):
        # Permute the indicies
        i_perm = self.get_permutation(indicies)
        # Ensure equal sizes of clusters
        i_perm = self.ensure_equal_sizes(i_perm, n_data)
        i_clusters = np.array_split(i_perm, self.n_clusters)
        return i_clusters

    def ensure_equal_sizes(self, i_perm, n_data, **kwargs):
        "Extend the permuted indicies so the clusters have equal sizes."
        # Find the number of points that should be added
        n_missing = (self.n_clusters * self.data_number) - n_data
        # Extend the permuted indicies
        if n_missing > 0:
            if n_missing > n_data:
                i_perm = np.append(
                    i_perm,
                    np.tile(i_perm, (n_missing // n_data) + 1)[:n_missing],
                )
            else:
                i_perm = np.append(i_perm, i_perm[:n_missing])
        return i_perm

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(data_number=self.data_number, seed=self.seed)
        # Get the constants made within the class
        constant_kwargs = dict(n_clusters=self.n_clusters)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
