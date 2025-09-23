from numpy import arange, argmin, empty
from .k_means import K_means


class FixedClustering(K_means):
    """
    Clustering algorithm class for data sets.
    It uses the distances to pre-defined fixed centroids for clustering.
    """

    def __init__(
        self,
        metric="euclidean",
        centroids=empty(0),
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the clustering algorithm.

        Parameters:
            metric: str
                The metric used to calculate the distances of the data.
            centroids: (K,D) array
                An array with the centroids of the K clusters.
                The centroids must have the same dimensions as the features.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set the arguments
        self.update_arguments(
            centroids=centroids,
            metric=metric,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def cluster_fit_data(self, X, **kwargs):
        indices = arange(len(X))
        i_min = argmin(self.calculate_distances(X, self.centroids), axis=1)
        return [indices[i_min == ki] for ki in range(self.n_clusters)]

    def update_arguments(
        self, metric=None, centroids=None, seed=None, dtype=None, **kwargs
    ):
        """
        Update the class with its arguments. The existing arguments are used
        if they are not given.

        Parameters:
            metric: str
                The metric used to calculate the distances of the data.
            centroids: (K,D) array
                An array with the centroids of the K clusters.
                The centroids must have the same dimensions as the features.
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
        if centroids is not None:
            self.set_centroids(centroids)
        if metric is not None:
            self.metric = metric
        # Set the parameters of the parent class
        super(K_means, self).update_arguments(
            seed=seed,
            dtype=dtype,
        )
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            metric=self.metric,
            centroids=self.centroids,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
