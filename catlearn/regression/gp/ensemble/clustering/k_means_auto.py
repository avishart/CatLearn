from numpy import append, arange, argmin, argsort, array, asarray
from numpy.linalg import norm
from .k_means import K_means


class K_means_auto(K_means):
    """
    Clustering algorithm class for data sets.
    It uses the K-means++ algorithm for clustering.
    It uses a interval of number of data points in each cluster.
    """

    def __init__(
        self,
        metric="euclidean",
        min_data=5,
        max_data=30,
        maxiter=100,
        tol=1e-4,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the clustering algorithm.

        Parameters:
            metric: str
                The metric used to calculate the distances of the data.
            min_data: int
                The minimum number of data point in each cluster.
            max_data: int
                The maximum number of data point in each cluster.
            maxiter: int
                The maximum number of iterations used to fit the clusters.
            tol: float
                The tolerance before the cluster fit is converged.
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
            min_data=min_data,
            max_data=max_data,
            maxiter=maxiter,
            tol=tol,
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
        centroids = self.initiate_centroids(X)
        # Optimize position of the centroids
        self.centroids, cluster_indices = self.optimize_centroids(
            X,
            centroids,
        )
        # Return the cluster indices
        return cluster_indices

    def update_arguments(
        self,
        metric=None,
        min_data=None,
        max_data=None,
        maxiter=None,
        tol=None,
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
            min_data: int
                The minimum number of data point in each cluster.
            max_data: int
                The maximum number of data point in each cluster.
            maxiter: int
                The maximum number of iterations used to fit the clusters.
            tol: float
                The tolerance before the cluster fit is converged.
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
        if min_data is not None:
            self.min_data = int(min_data)
        if max_data is not None:
            self.max_data = int(max_data)
        # Check that the numbers of used data points agree
        if self.max_data < self.min_data:
            self.max_data = int(self.min_data)
        # Set the arguments of the parent class
        super().update_arguments(
            metric=metric,
            n_clusters=None,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            dtype=dtype,
        )
        return self

    def calc_n_clusters(self, X, **kwargs):
        """
        Calculate the number of clusters based on the data.
        """
        n_data = len(X)
        n_clusters = int(n_data // self.max_data)
        if n_data > (n_clusters * self.max_data):
            n_clusters += 1
        return n_clusters

    def optimize_centroids(self, X, centroids, **kwargs):
        "Optimize the positions of the centroids."
        indices = arange(len(X))
        for _ in range(1, self.maxiter + 1):
            # Store the old centroids
            centroids_old = centroids.copy()
            # Calculate which centroids that are closest
            distance_matrix = self.calculate_distances(X, centroids)
            cluster_indices = self.count_clusters(
                X,
                indices,
                distance_matrix,
            )
            centroids = asarray(
                [X[indices_ki].mean(axis=0) for indices_ki in cluster_indices]
            )
            # Check if it is converged
            if norm(centroids - centroids_old) <= self.tol:
                break
        return centroids, cluster_indices

    def count_clusters(self, X, indices, distance_matrix, **kwargs):
        """
        Get the indices for each of the clusters.
        The number of data points in each cluster is counted and restricted
        between the minimum and maximum number of allowed cluster sizes.
        """
        # Make a list cluster indices
        klist = arange(self.n_clusters).reshape(-1, 1)
        # Find the cluster that each point is closest to
        k_indices = argmin(distance_matrix, axis=1)
        indices_ki_bool = klist == k_indices
        # Check the number of points per cluster
        n_ki = indices_ki_bool.sum(axis=1)
        # Ensure the number is within the conditions
        n_ki[n_ki > self.max_data] = self.max_data
        n_ki[n_ki < self.min_data] = self.min_data
        # Sort the indices as function of the distances to the centroids
        d_indices = argsort(distance_matrix, axis=0)
        indices_sorted = indices[d_indices.T]
        indices_ki_bool = indices_ki_bool[klist, indices_sorted]
        # Prioritize the points that is part of each cluster
        cluster_indices = [
            append(
                indices_sorted[ki, indices_ki_bool[ki]],
                indices_sorted[ki, ~indices_ki_bool[ki]],
            )[: n_ki[ki]]
            for ki in range(self.n_clusters)
        ]
        return cluster_indices

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            metric=self.metric,
            min_data=self.min_data,
            max_data=self.max_data,
            maxiter=self.maxiter,
            tol=self.tol,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict(n_clusters=self.n_clusters)
        # Get the objects made within the class
        object_kwargs = dict(centroids=self.centroids)
        return arg_kwargs, constant_kwargs, object_kwargs
