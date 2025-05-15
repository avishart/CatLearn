from numpy import asarray, exp
import pickle
import warnings
from ..means.constant import Prior_constant
from ..calculator.mlmodel import get_default_model


class EnsembleModel:
    def __init__(
        self,
        model=None,
        use_variance_ensemble=True,
        use_softmax=False,
        use_same_prior_mean=True,
        dtype=float,
        **kwargs,
    ):
        """
        Ensemble model of machine learning models.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
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
        # Set the arguments
        self.update_arguments(
            model=model,
            use_variance_ensemble=use_variance_ensemble,
            use_softmax=use_softmax,
            use_same_prior_mean=use_same_prior_mean,
            dtype=dtype,
            **kwargs,
        )

    def train(self, features, targets, **kwargs):
        """
        Train the model with training features and targets.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array or (N,1+D) array
                Training targets with N data points.
                If use_derivatives=True, the training targets is in
                first column and derivatives is in the next columns.

        Returns:
            self: The trained model object itself.
        """
        raise NotImplementedError()

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
        """
        Optimize the hyperparameter of the model and its kernel.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array or (N,D+1) array
                Training targets with or without derivatives with
                N data points.
            retrain: bool
                Whether to retrain the model after the optimization.
            hp: dict
                Use a set of hyperparameters to optimize from
                else the current set is used.
            maxiter: int
                Maximum number of iterations used by local or
                global optimization method.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            verbose: bool
                Print the optimized hyperparameters and
                the object function value.

        Returns:
            list: List of solution dictionaries with objective function value,
                optimized hyperparameters, success statement,
                and number of used evaluations.
        """
        raise NotImplementedError()

    def predict(
        self,
        features,
        get_derivatives=False,
        get_variance=False,
        include_noise=False,
        get_derivtives_var=False,
        get_var_derivatives=False,
        **kwargs,
    ):
        """
        Predict the mean and variance for test features by using data and
        coefficients from training data.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            get_derivatives: bool
                Whether to predict the derivatives of the prediction mean.
            get_variance: bool
                Whether to predict the variance of the targets.
            include_noise: bool
                Whether to include the noise of data in the predicted variance.
            get_derivtives_var: bool
                Whether to predict the variance of the derivatives
                of the targets.
            get_var_derivatives: bool
                Whether to calculate the derivatives of the predicted variance
                of the targets.

        Returns:
            Y_predict: (M,1) or (M,1+D) array
                The predicted mean values with or without derivatives.
            var: (M,1) or (M,1+D) array
                The predicted variance of the targets with or
                without derivatives.
            var_deriv: (M,D) array
                The derivatives of the predicted variance of the targets.
        """
        # Calculate the predicted values for one model
        if self.n_models == 1:
            return self.model_prediction(
                self.model,
                features,
                get_derivatives=get_derivatives,
                get_variance=get_variance,
                include_noise=include_noise,
                get_derivtives_var=get_derivtives_var,
                get_var_derivatives=get_var_derivatives,
                **kwargs,
            )
        # Ensure the right arguments are chosen
        if self.use_variance_ensemble:
            get_variance = True
            if get_var_derivatives or get_derivtives_var:
                get_derivatives = True
            if get_derivatives:
                get_var_derivatives = True
        if get_var_derivatives:
            get_derivatives = True
        # Calculate the predicted values for multiple model
        Y_preds = []
        var_preds = []
        var_derivs = []
        for model in self.models:
            Y_predict, var, var_deriv = self.model_prediction(
                model,
                features,
                get_derivatives=get_derivatives,
                get_variance=get_variance,
                include_noise=include_noise,
                get_derivtives_var=get_derivtives_var,
                get_var_derivatives=get_var_derivatives,
                **kwargs,
            )
            Y_preds.append(Y_predict)
            var_preds.append(var)
            var_derivs.append(var_deriv)
        return self.ensemble(
            Y_preds,
            var_preds,
            var_derivs,
            get_derivatives=get_derivatives,
            get_variance=get_variance,
            get_derivtives_var=get_derivtives_var,
            get_var_derivatives=get_var_derivatives,
        )

    def predict_mean(self, features, get_derivatives=False, **kwargs):
        """
        Predict the mean for test features by using data and coefficients
        from training data.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            get_derivatives: bool
                Whether to predict the derivatives of the prediction mean.

        Returns:
            Y_predict: (M,1) array
                The predicted mean values if get_derivatives=False
            or
            Y_predict: (M,1+D) array
                The predicted mean values and its derivatives
                if get_derivatives=True.
        """
        # Check if the variance was needed for prediction mean
        if self.use_variance_ensemble:
            raise AttributeError(
                "The predict_mean function is not defined "
                "with use_variance_ensemble=True!"
            )
        # Calculate the predicted values for one model
        if self.n_models == 1:
            return self.model_prediction_mean(
                self.model,
                features,
                get_derivatives=get_derivatives,
                **kwargs,
            )
        # Calculate the predicted values for multiple model
        Y_preds = []
        for model in self.models:
            Y_predict = self.model_prediction_mean(
                model,
                features,
                get_derivatives=get_derivatives,
                **kwargs,
            )
            Y_preds.append(Y_predict)
        return self.ensemble(
            Y_preds,
            get_derivatives=get_derivatives,
            get_variance=False,
        )

    def predict_variance(
        self,
        features,
        get_derivatives=False,
        include_noise=False,
        **kwargs,
    ):
        """
        Calculate the predicted variance of the test targets.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features.
                If KQX=None, it is calculated.
            get_derivatives: bool
                Whether to predict the uncertainty of the derivatives
                of the targets.
            include_noise: bool
                Whether to include the noise of data in the predicted variance

        Returns:
            var: (M,1) array
                The predicted variance of the targets if get_derivatives=False.
            or
            var: (M,1+D) array
                The predicted variance of the targets and its derivatives
                if get_derivatives=True.

        """
        raise NotImplementedError()

    def calculate_variance_derivatives(self, features, **kwargs):
        """
        Calculate the derivatives of the predicted variance
        of the test targets.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features.
                If KQX=None, it is calculated.

        Returns:
            var_deriv: (M,D) array
                The derivatives of the predicted variance of the targets.
        """
        raise NotImplementedError()

    def get_hyperparams(self, **kwargs):
        """
        Get the hyperparameters for the model and the kernel.

        Returns:
            dict: The hyperparameters in the log-space from
                the model and kernel class if multiple models are not defined.
            or
            list: A list of dictionaries with the hyperparameters
                in the log-space from the model and kernel class
                if multiple models are defined.
        """
        if len(self.models):
            return [model.get_hyperparams() for model in self.models]
        return self.model.get_hyperparams()

    def get_use_derivatives(self):
        "Get whether the derivatives of the targets are used."
        return self.model.get_use_derivatives()

    def get_use_fingerprint(self):
        "Get whether a fingerprint is used as the features."
        return self.model.get_use_fingerprint()

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the data type
        self.dtype = dtype
        # Set the data type of the attributes
        self.model.set_dtype(dtype=dtype, **kwargs)
        self.copy_prior()
        if len(self.models):
            for model in self.models:
                model.set_dtype(dtype=dtype, **kwargs)
        return self

    def set_seed(self, seed, **kwargs):
        """
        Set the random seed.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        self.model.set_seed(seed)
        if len(self.models):
            for model in self.models:
                model.set_seed(seed)
        return self

    def set_use_derivatives(self, use_derivatives, **kwargs):
        """
        Set whether to use derivatives/gradients for training and predictions.

        Parameters:
            use_derivatives: bool
                Use derivatives/gradients for training and predictions.

        Returns:
            self: The updated object itself.
        """
        # Set whether to use derivatives for the kernel
        self.model.set_use_derivatives(use_derivatives)
        return self

    def set_use_fingerprint(self, use_fingerprint, **kwargs):
        """
        Set whether to use a fingerprint as the features.

        Parameters:
            use_fingerprint: bool
                Use a fingerprint as the features.

        Returns:
            self: The updated object itself.
        """
        # Set whether to use a fingerprint for the features
        self.model.set_use_fingerprint(use_fingerprint)
        return self

    def save_model(self, filename="model.pkl", **kwargs):
        """
        Save the model object to a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            self: The object itself.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return self

    def load_model(self, filename="model.pkl", **kwargs):
        """
        Load the model object from a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            model: The loaded model object.
        """
        with open(filename, "rb") as file:
            model = pickle.load(file)
        return model

    def update_arguments(
        self,
        model=None,
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
        if model is not None:
            self.model = model.copy()
            # Set descriptor of the ensemble model
            self.reset_models()
        if use_variance_ensemble is not None:
            self.use_variance_ensemble = use_variance_ensemble
        if use_softmax is not None:
            self.use_softmax = use_softmax
        if use_same_prior_mean is not None:
            self.use_same_prior_mean = use_same_prior_mean
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype=dtype, **kwargs)
        return self

    def copy_prior(self):
        "Copy the prior of the model."
        self.prior = self.model.prior.copy()
        return self

    def reset_models(self, **kwargs):
        "Reset the models."
        # Set descriptor of the ensemble model
        self.n_models = 1
        self.models = []
        # Get the prior mean instance
        self.copy_prior()
        return self

    def model_training(self, model, features, targets, **kwargs):
        "Train the model."
        return model.train(features, targets, **kwargs)

    def model_optimization(
        self,
        model,
        features,
        targets,
        retrain=True,
        hp=None,
        pdis=None,
        verbose=False,
        **kwargs,
    ):
        "Optimize the hyperparameters of the model."
        return model.optimize(
            features,
            targets,
            retrain=retrain,
            hp=hp,
            pdis=pdis,
            verbose=verbose,
            **kwargs,
        )

    def model_prediction(
        self,
        model,
        features,
        get_derivatives=False,
        get_variance=False,
        include_noise=False,
        get_derivtives_var=False,
        get_var_derivatives=False,
        **kwargs,
    ):
        "Predict mean, variance, and variance derivatives with the model."
        return model.predict(
            features,
            get_derivatives=get_derivatives,
            get_variance=get_variance,
            include_noise=include_noise,
            get_derivtives_var=get_derivtives_var,
            get_var_derivatives=get_var_derivatives,
            **kwargs,
        )

    def model_prediction_mean(
        self,
        model,
        features,
        get_derivatives=False,
        **kwargs,
    ):
        "Predict mean with the model."
        return model.predict_mean(
            features,
            get_derivatives=get_derivatives,
            **kwargs,
        )

    def model_prediction_variance(
        self,
        model,
        features,
        get_derivatives=False,
        include_noise=False,
        **kwargs,
    ):
        "Predict variance with the model."
        return model.predict_variance(
            features,
            get_derivatives=get_derivatives,
            include_noise=include_noise,
            **kwargs,
        )

    def model_variance_derivatives(self, model, features, **kwargs):
        """
        Calculate the derivatives of the predicted variance
        of the test targets.
        """
        return model.calculate_variance_derivatives(features, **kwargs)

    def ensemble(
        self,
        Y_preds,
        var_preds=None,
        var_derivs=None,
        get_derivatives=False,
        get_variance=True,
        get_derivtives_var=False,
        get_var_derivatives=False,
        **kwargs,
    ):
        """
        Make an ensemble of the predicted values and variances.
        The variance weighted ensemble is used if variance_ensemble=True.
        """
        # Transform the input to arrays
        Y_preds = asarray(Y_preds)
        if get_variance:
            var_preds = asarray(var_preds)
        else:
            var_preds = None
        if get_var_derivatives and var_derivs is not None:
            var_derivs = asarray(var_derivs)
        else:
            var_derivs = None
        # Perform ensemble of the predictions
        if self.use_variance_ensemble:
            return self.ensemble_variance(
                Y_preds,
                var_preds=var_preds,
                var_derivs=var_derivs,
                get_derivatives=get_derivatives,
                get_variance=get_variance,
                get_derivtives_var=get_derivtives_var,
                get_var_derivatives=get_var_derivatives,
                **kwargs,
            )
        return self.ensemble_mean(
            Y_preds,
            var_preds=var_preds,
            var_derivs=var_derivs,
            get_derivatives=get_derivatives,
            get_variance=get_variance,
            get_derivtives_var=get_derivtives_var,
            get_var_derivatives=get_var_derivatives,
            **kwargs,
        )

    def ensemble_mean(
        self,
        Y_preds,
        var_preds=None,
        var_derivs=None,
        get_derivatives=False,
        get_variance=True,
        get_derivtives_var=False,
        get_var_derivatives=False,
        **kwargs,
    ):
        """
        Make an ensemble of the predicted values and variances.
        The average is to weight predictions.
        """
        # Default predictions
        var_predict = None
        var_deriv = None
        # Calculate the prediction mean
        Y_predict = Y_preds.mean(axis=0)
        # Calculate the predicted variance
        if get_variance:
            var_predict = (var_preds + ((Y_preds - Y_predict) ** 2)).mean(
                axis=0
            )
        # Calculate the derivative of the predicted variance
        if get_var_derivatives:
            var_deriv = (
                var_derivs
                + (
                    2.0
                    * (Y_preds[:, :, 0] - Y_predict[:, 0])
                    * (Y_preds[:, :, 1:] - Y_predict[:, 1:])
                )
            ).mean(axis=0)
        return Y_predict, var_predict, var_deriv

    def ensemble_variance(
        self,
        Y_preds,
        var_preds=None,
        var_derivs=None,
        get_derivatives=False,
        get_variance=True,
        get_derivtives_var=False,
        get_var_derivatives=False,
        **kwargs,
    ):
        """
        Make an ensemble of the predicted values and variances.
        The variance weighted ensemble is used.
        """
        # Default predictions
        var_predict = None
        var_deriv = None
        # Calculate the weights
        weights, weights_deriv = self.get_weights(
            var_preds, var_derivs, get_derivatives
        )
        # Calculate the prediction mean
        Y_predict = (weights * Y_preds).sum(axis=0)
        # Calculate the derivative of the prediction mean
        if get_derivatives:
            # Add extra contribution from weight derivatives
            Y_predict[:, 1:] += (Y_preds[:, :, 0:1] * weights_deriv).sum(
                axis=0
            )
        # Calculate the predicted variance
        if get_variance:
            var_predict = (
                weights * (var_preds + ((Y_preds - Y_predict) ** 2))
            ).sum(axis=0)
            if get_derivtives_var:
                warnings.warn(
                    "Check if it is the right expression for"
                    "the variance of the derivatives!"
                )
        # Calculate the derivative of the predicted variance
        if get_var_derivatives:
            var_deriv = (
                weights
                * (
                    var_derivs
                    + (
                        2.0
                        * (Y_preds[:, :, 0:1] - Y_predict[:, 0:1])
                        * (Y_preds[:, :, 1:] - Y_predict[:, 1:])
                    )
                )
            ).sum(axis=0)
            var_deriv += (var_preds[:, :, 0:1] * weights_deriv).sum(axis=0)
        return Y_predict, var_predict, var_deriv

    def get_weights(
        self,
        var_preds=None,
        var_derivs=None,
        get_derivatives=False,
        **kwargs,
    ):
        "Calculate the weights."
        weights_deriv = None
        if var_preds is None:
            raise AttributeError("The predicted variance is missing!")
        # Use the predicted variance to weight predictions
        if self.use_softmax:
            var_coef = exp(-var_preds[:, :, 0:1])
        else:
            var_coef = 1.0 / var_preds[:, :, 0:1]
        # Normalize the weights
        weights = var_coef / var_coef.sum(axis=0)
        # Calculate the derivative of the prediction mean
        if get_derivatives:
            # Calculate the derivative of the weights
            if self.use_softmax:
                weights_deriv = weights * (
                    (weights * var_derivs).sum(axis=0) - var_derivs
                )
            else:
                weights_deriv = weights * (
                    (weights * var_coef * var_derivs).sum(axis=0)
                    - (var_coef * var_derivs)
                )
        return weights, weights_deriv

    def get_models(self, **kwargs):
        "Get the models."
        return [model.copy() for model in self.models]

    def set_same_prior_mean(self, model, features, targets, **kwargs):
        "Set the same prior mean constant for the models."
        if self.use_same_prior_mean:
            self.prior.update(features, targets, **kwargs)
            prior_parameters = self.prior.get_parameters()
            model.update_arguments(prior=Prior_constant(**prior_parameters))
        return model

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            model=self.model,
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

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)
