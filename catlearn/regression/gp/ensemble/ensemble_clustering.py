from numpy import array, ndarray
from .ensemble import EnsembleModel, get_default_model
from .clustering.k_means_number import K_means_number


class EnsembleClustering(EnsembleModel):
    """
    Ensemble model of machine learning models.
    The ensemble model is used to combine the predictions
    of multiple machine learning models.
    The ensemle models are chosen by a clustering algorithm.
    """

    def __init__(
        self,
        model=None,
        clustering=None,
        use_variance_ensemble=True,
        use_softmax=False,
        use_same_prior_mean=True,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the ensemble model.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
            clustering: Clustering class object
                The clustering method used to split the data
                to different models.
            use_variance_ensemble: bool
                Whether to use the predicted inverse variances
                to weight the predictions.
                Else an average of the predictions is used.
            use_softmax: bool
                Whether to use the softmax of the predicted inverse variances
                as weights.
                It is only active if use_variance_ensemble=True, too.
            use_same_prior_mean: bool
                Whether to use the same prior mean for all models.
            dtype: type
                The data type of the arrays.
        """
        # Make default model if it is not given
        if model is None:
            model = get_default_model(dtype=dtype)
        # Make default clustering if it is not given
        if clustering is None:
            clustering = K_means_number(dtype=dtype)
        # Set the arguments
        self.update_arguments(
            model=model,
            clustering=clustering,
            use_variance_ensemble=use_variance_ensemble,
            use_softmax=use_softmax,
            use_same_prior_mean=use_same_prior_mean,
            dtype=dtype,
            **kwargs,
        )

    def train(self, features, targets, **kwargs):
        # Cluster the data
        cdata = self.cluster(features, targets)
        # Set the number of models
        self.n_models = len(cdata)
        self.models = []
        # Set the same prior mean to all models if specified
        self.set_same_prior_mean(self.model, features, targets)
        # If only one model is used
        if self.n_models == 1:
            self.model_training(self.model, features, targets, **kwargs)
            self.models.append(self.model)
            return self
        # If multiple models are used
        for ki in range(self.n_models):
            model = self.model.copy()
            self.model_training(model, cdata[ki][0], cdata[ki][1], **kwargs)
            self.models.append(model)
        return self

    def optimize(
        self,
        features,
        targets,
        retrain=True,
        hp=None,
        pdis=None,
        verbose=False,
        **kwargs,
    ):
        # Cluster the data
        cdata = self.cluster(features, targets)
        # Set the number of models
        self.n_models = len(cdata)
        self.models = []
        sols = []
        # Set the same prior mean to all models if specified
        self.set_same_prior_mean(self.model, features, targets)
        # If only one model is used
        if self.n_models == 1:
            sol = self.model_optimization(
                self.model,
                features,
                targets,
                retrain=retrain,
                hp=hp,
                pdis=pdis,
                verbose=verbose,
                **kwargs,
            )
            sols.append(sol)
            self.models.append(self.model)
            return sols
        # If multiple models are used
        for ki in range(self.n_models):
            model = self.model.copy()
            sol = self.model_optimization(
                model,
                cdata[ki][0],
                cdata[ki][1],
                retrain=retrain,
                hp=hp,
                pdis=pdis,
                verbose=verbose,
                **kwargs,
            )
            sols.append(sol)
            self.models.append(model)
        return sols

    def set_dtype(self, dtype, **kwargs):
        super().set_dtype(dtype, **kwargs)
        # Set the data type of the clustering
        self.clustering.set_dtype(dtype=dtype)
        return self

    def set_seed(self, seed, **kwargs):
        super().set_seed(seed, **kwargs)
        # Set the random seed of the clustering
        self.clustering.set_seed(seed=seed)
        return self

    def update_arguments(
        self,
        model=None,
        clustering=None,
        use_variance_ensemble=None,
        use_softmax=None,
        use_same_prior_mean=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
            clustering: Clustering class object
                The clustering method used to split the data
                to different models.
            use_variance_ensemble: bool
                Whether to use the predicted inverse variances
                to weight the predictions.
                Else an average of the predictions is used.
            use_softmax: bool
                Whether to use the softmax of the predicted inverse variances
                as weights.
                It is only active if use_variance_ensemble=True, too.
            use_same_prior_mean: bool
                Whether to use the same prior mean for all models.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        if clustering is not None:
            self.clustering = clustering.copy()
        # Set the parameters for the parent class
        super().update_arguments(
            model=model,
            use_variance_ensemble=use_variance_ensemble,
            use_softmax=use_softmax,
            use_same_prior_mean=use_same_prior_mean,
            dtype=dtype,
        )
        return self

    def cluster(self, features, targets, **kwargs):
        "Cluster the data."
        if isinstance(features[0], (ndarray, list)):
            X = array(features, dtype=self.dtype)
        else:
            X = array(
                [feature.get_vector() for feature in features],
                dtype=self.dtype,
            )
        cluster_indicies = self.clustering.cluster_fit_data(X)
        return [
            (features[indicies_ki], targets[indicies_ki])
            for indicies_ki in cluster_indicies
        ]

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            model=self.model,
            clustering=self.clustering,
            use_variance_ensemble=self.use_variance_ensemble,
            use_softmax=self.use_softmax,
            use_same_prior_mean=self.use_same_prior_mean,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict(n_models=self.n_models)
        # Get the objects made within the class
        object_kwargs = dict(models=self.get_models())
        return arg_kwargs, constant_kwargs, object_kwargs
