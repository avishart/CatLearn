from numpy import append, arange, argmin, argsort, array, asarray
from numpy.linalg import norm
from .k_means import K_means


class K_means_number(K_means):
    def __init__(
        self,
        metric="euclidean",
        data_number=25,
        maxiter=100,
        tol=1e-4,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Clustering class object for data sets.
        The K-means++ algorithm for clustering, but where the number
        of clusters are updated from a fixed number data point in each cluster.

        Parameters:
            metric: str
                The metric used to calculate the distances of the data.
            data_number: int
                The number of data point in each cluster.
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
            data_number=data_number,
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
        # If only one cluster is used give the full data
        if self.n_clusters == 1:
            self.centroids = asarray([X.mean(axis=0)])
            return [arange(len(X))]
        # Initiate the centroids
        centroids = self.initiate_centroids(X)
        # Optimize position of the centroids
        self.centroids, cluster_indicies = self.optimize_centroids(
            X,
            centroids,
        )
        # Return the cluster indicies
        return cluster_indicies

    def update_arguments(
        self,
        metric=None,
        data_number=None,
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
            data_number: int
                The number of data point in each cluster.
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
        if data_number is not None:
            self.data_number = int(data_number)
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
        n_clusters = int(n_data // self.data_number)
        if n_data - (n_clusters * self.data_number):
            n_clusters += 1
        return n_clusters

    def optimize_centroids(self, X, centroids, **kwargs):
        "Optimize the positions of the centroids."
        indicies = arange(len(X))
        for _ in range(1, self.maxiter + 1):
            # Store the old centroids
            centroids_old = centroids.copy()
            # Calculate which centroids that are closest
            distance_matrix = self.calculate_distances(X, centroids)
            cluster_indicies = self.count_clusters(
                X,
                indicies,
                distance_matrix,
            )
            centroids = asarray(
                [
                    X[indicies_ki].mean(axis=0)
                    for indicies_ki in cluster_indicies
                ]
            )
            # Check if it is converged
            if norm(centroids - centroids_old) <= self.tol:
                break
        return centroids, cluster_indicies

    def count_clusters(self, X, indicies, distance_matrix, **kwargs):
        """
        Get the indicies for each of the clusters.
        The number of data points in each cluster is counted and restricted
        between the minimum and maximum number of allowed cluster sizes.
        """
        # Make a list cluster indicies
        klist = arange(self.n_clusters).reshape(-1, 1)
        # Find the cluster that each point is closest to
        k_indicies = argmin(distance_matrix, axis=1)
        indicies_ki_bool = klist == k_indicies
        # Sort the indicies as function of the distances to the centroids
        d_indicies = argsort(distance_matrix, axis=0)
        indicies_sorted = indicies[d_indicies.T]
        indicies_ki_bool = indicies_ki_bool[klist, indicies_sorted]
        # Prioritize the points that is part of each cluster
        cluster_indicies = [
            append(
                indicies_sorted[ki, indicies_ki_bool[ki]],
                indicies_sorted[ki, ~indicies_ki_bool[ki]],
            )[: self.data_number]
            for ki in range(self.n_clusters)
        ]
        return cluster_indicies

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            metric=self.metric,
            data_number=self.data_number,
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
