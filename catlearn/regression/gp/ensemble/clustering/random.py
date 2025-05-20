from numpy import append, arange, array_split, tile
from .clustering import Clustering


class RandomClustering(Clustering):
    """
    Clustering algorithn class for data sets.
    It uses randomized clusters for clustering.
    """

    def __init__(
        self,
        n_clusters=4,
        equal_size=True,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the clustering algorithm.

        Parameters:
            n_clusters: int
                The number of used clusters.
            equal_size: bool
                Whether the clusters are forced to have the same size.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            n_clusters=n_clusters,
            equal_size=equal_size,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def cluster_fit_data(self, X, **kwargs):
        # Make indicies
        n_data = len(X)
        indicies = arange(n_data)
        # If only one cluster is used give the full data
        if self.n_clusters == 1:
            return [indicies]
        # Randomly make clusters
        i_clusters = self.randomized_clusters(indicies, n_data)
        # Return the cluster indicies
        return i_clusters

    def cluster(self, X, **kwargs):
        return self.cluster_fit_data(X)

    def update_arguments(
        self,
        n_clusters=None,
        equal_size=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            n_clusters: int
                The number of used clusters.
            equal_size: bool
                Whether the clusters are forced to have the same size.
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
        if n_clusters is not None:
            self.n_clusters = int(n_clusters)
        if equal_size is not None:
            self.equal_size = equal_size
        # Set the parameters of the parent class
        super().update_arguments(
            seed=seed,
            dtype=dtype,
        )
        return self

    def randomized_clusters(self, indicies, n_data, **kwargs):
        "Randomized indicies used for each cluster."
        # Permute the indicies
        i_perm = self.get_permutation(indicies)
        # Ensure equal sizes of clusters if chosen
        if self.equal_size:
            i_perm = self.ensure_equal_sizes(i_perm, n_data)
        i_clusters = array_split(i_perm, self.n_clusters)
        return i_clusters

    def get_permutation(self, indicies):
        "Permute the indicies"
        return self.rng.permutation(indicies)

    def ensure_equal_sizes(self, i_perm, n_data, **kwargs):
        "Extend the permuted indicies so the clusters have equal sizes."
        # Find the number of excess points left
        n_left = n_data % self.n_clusters
        # Find the number of points that should be added
        if n_left > 0:
            n_missing = self.n_clusters - n_left
        else:
            n_missing = 0
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
            n_clusters=self.n_clusters,
            equal_size=self.equal_size,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
