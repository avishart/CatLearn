from numpy import append, arange, array, asarray
from .k_means import K_means


class K_means_enumeration(K_means):
    """
    Clustering algorithm class for data sets.
    It uses the K-means++ algorithm for clustering.
    It uses a fixed number of data points in each cluster.
    """

    def __init__(
        self,
        metric="euclidean",
        data_number=25,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the clustering algorithm.

        Parameters:
            metric: str
                The metric used to calculate the distances of the data.
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
        super().__init__(
            metric=metric,
            data_number=data_number,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def cluster_fit_data(self, X, **kwargs):
        # Copy the data
        X = array(X, dtype=self.dtype)
        # Calculate the number of clusters
        self.n_clusters = self.calc_n_clusters(X)
        # If only one cluster is used, give the full data
        if self.n_clusters == 1:
            self.centroids = asarray([X.mean(axis=0)])
            return [arange(len(X))]
        # Initiate the centroids
        self.centroids, cluster_indices = self.initiate_centroids(X)
        # Return the cluster indices
        return cluster_indices

    def update_arguments(
        self,
        metric=None,
        data_number=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            metric: str
                The metric used to calculate the distances of the data.
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
        # Set the arguments of the parent class
        super().update_arguments(
            metric=metric,
            n_clusters=None,
            seed=seed,
            dtype=dtype,
        )
        return self

    def calc_n_clusters(self, X, **kwargs):
        """
        Calculate the number of clusters based on the data.
        """
        n_data = len(X)
        n_clusters = int(n_data // self.data_number)
        if n_data - (n_clusters * self.data_number):
            n_clusters += 1
        return n_clusters

    def initiate_centroids(self, X, **kwargs):
        "Initial the centroids from the K-mean++ method."
        n_data = len(X)
        indices = arange(n_data)
        if int(self.n_clusters * self.data_number) > n_data:
            n_f = int((self.n_clusters - 1) * self.data_number)
            n_r = int(n_data - self.data_number)
            indices = append(indices[:n_f], indices[n_r:])
        indices = indices.reshape(self.n_clusters, self.data_number)
        centroids = asarray(
            [X[indices_ki].mean(axis=0) for indices_ki in indices]
        )
        return centroids, indices

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            metric=self.metric,
            data_number=self.data_number,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict(n_clusters=self.n_clusters)
        # Get the objects made within the class
        object_kwargs = dict(centroids=self.centroids)
        return arg_kwargs, constant_kwargs, object_kwargs
