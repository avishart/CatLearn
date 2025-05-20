from numpy import append, arange, array_split, tile
from .random import RandomClustering


class RandomClustering_number(RandomClustering):
    """
    Clustering algorithn class for data sets.
    It uses randomized clusters for clustering.
    It uses a fixed number of data points in each cluster.
    """

    def __init__(self, data_number=25, seed=None, dtype=float, **kwargs):
        """
        Initialize the clustering algorithm.

        Parameters:
            data_number: int
                The number of data point in each cluster.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super(RandomClustering, self).__init__(
            data_number=data_number,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def cluster_fit_data(self, X, **kwargs):
        # Make indicies
        n_data = len(X)
        indicies = arange(n_data)
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

    def update_arguments(
        self,
        data_number=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments. The existing arguments are used
        if they are not given.

        Parameters:
            data_number: int
                The number of data point in each cluster.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        if data_number is not None:
            self.data_number = int(data_number)
        # Set the parameters of the parent class
        super(RandomClustering, self).update_arguments(
            seed=seed,
            dtype=dtype,
        )
        return self

    def randomized_clusters(self, indicies, n_data, **kwargs):
        # Permute the indicies
        i_perm = self.get_permutation(indicies)
        # Ensure equal sizes of clusters
        i_perm = self.ensure_equal_sizes(i_perm, n_data)
        i_clusters = array_split(i_perm, self.n_clusters)
        return i_clusters

    def ensure_equal_sizes(self, i_perm, n_data, **kwargs):
        "Extend the permuted indicies so the clusters have equal sizes."
        # Find the number of points that should be added
        n_missing = (self.n_clusters * self.data_number) - n_data
        # Extend the permuted indicies
        if n_missing > 0:
            if n_missing > n_data:
                i_perm = append(
                    i_perm,
                    tile(i_perm, (n_missing // n_data) + 1)[:n_missing],
                )
            else:
                i_perm = append(i_perm, i_perm[:n_missing])
        return i_perm

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            data_number=self.data_number,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict(n_clusters=self.n_clusters)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
