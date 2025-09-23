from numpy import array, asarray, full, log, sqrt
from scipy.spatial.distance import pdist
from .restricted import RestrictedBoundaries


class EducatedBoundaries(RestrictedBoundaries):
    """
    Boundary conditions for the hyperparameters with educated guess for
    the length-scale, relative-noise, and prefactor hyperparameters.
    Machine precisions are used as boundary conditions for
    other hyperparameters not given in the dictionary.
    """

    def __init__(
        self,
        bounds_dict={},
        scale=1.0,
        use_log=True,
        max_length=True,
        use_derivatives=False,
        use_prior_mean=True,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the boundary conditions for the hyperparameters.

        Parameters:
            bounds_dict: dict
                A dictionary with boundary conditions as numpy (H,2) arrays
                with two columns for each type of hyperparameter.
            scale: float
                Scale the boundary conditions.
            use_log: bool
                Whether to use hyperparameters in log-scale or not.
            max_length: bool
                Whether to use the maximum scaling for the length-scale or
                use a more reasonable scaling.
            use_derivatives: bool
                Whether the derivatives of the target are used in the model.
                The boundary conditions of the length-scale hyperparameter(s)
                will change with the use_derivatives.
                The use_derivatives will be updated
                when update_bounds is called.
            use_prior_mean: bool
                Whether to use the prior mean to calculate the boundary of
                the prefactor hyperparameter.
                If use_prior_mean=False, the minimum and maximum target
                differences are used as the boundary conditions.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        self.update_arguments(
            bounds_dict=bounds_dict,
            scale=scale,
            use_log=use_log,
            max_length=max_length,
            use_derivatives=use_derivatives,
            use_prior_mean=use_prior_mean,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def update_arguments(
        self,
        bounds_dict=None,
        scale=None,
        use_log=None,
        max_length=None,
        use_derivatives=None,
        use_prior_mean=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            bounds_dict: dict
                A dictionary with boundary conditions as numpy (H,2) arrays
                with two columns for each type of hyperparameter.
            scale: float
                Scale the boundary conditions.
            use_log: bool
                Whether to use hyperparameters in log-scale or not.
            max_length: bool
                Whether to use the maximum scaling for the length-scale or
                use a more reasonable scaling.
            use_derivatives: bool
                Whether the derivatives of the target are used in the model.
                The boundary conditions of the length-scale hyperparameter(s)
                will change with the use_derivatives.
                The use_derivatives will be updated
                when update_bounds is called.
            use_prior_mean: bool
                Whether to use the prior mean to calculate the boundary of
                the prefactor hyperparameter.
                If use_prior_mean=False, the minimum and maximum target
                differences are used as the boundary conditions.
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
        # Update the parameters of the parent class
        super().update_arguments(
            bounds_dict=bounds_dict,
            scale=scale,
            use_log=use_log,
            max_length=max_length,
            use_derivatives=use_derivatives,
            seed=seed,
            dtype=dtype,
        )
        # Update the parameters of the class itself
        if use_prior_mean is not None:
            self.use_prior_mean = use_prior_mean
        return self

    def make_bounds(self, model, X, Y, parameters, parameters_set, **kwargs):
        eps_lower, eps_upper = self.get_boundary_limits()
        self.get_use_derivatives(model)
        bounds = {}
        for para in parameters_set:
            if para == "length":
                bounds[para] = self.length_bound(X, parameters.count(para))
            elif para == "noise":
                if "noise_deriv" in parameters_set:
                    bounds[para] = self.noise_bound(
                        Y[:, 0:1],
                        eps_lower=eps_lower,
                    )
                else:
                    bounds[para] = self.noise_bound(Y, eps_lower=eps_lower)
            elif para == "noise_deriv":
                bounds[para] = self.noise_bound(Y[:, 1:], eps_lower=eps_lower)
            elif para == "prefactor":
                bounds[para] = self.prefactor_bound(X, Y, model)
            elif para in self.bounds_dict:
                bounds[para] = array(self.bounds_dict[para], dtype=self.dtype)
            else:
                bounds[para] = full(
                    (parameters.count(para), 2),
                    [eps_lower, eps_upper],
                    dtype=self.dtype,
                )
        return bounds

    def prefactor_bound(self, X, Y, model, **kwargs):
        """
        Get the minimum and maximum ranges of the prefactor
        in the educated guess regime within a scale.
        """
        if self.use_prior_mean:
            # Get the prior mean value for the target only
            Y_mean = self.get_prior_mean(X, Y, model)
            Y_std = Y[:, 0:1] - Y_mean
            # Calculate the variance relative to the prior mean of the targets
            a_mean = sqrt((Y_std**2).mean())
            # Check that all the targets are not the same
            if a_mean == 0.0:
                a_mean = 1.00
            scaling = 10.0 * self.scale
            a_max = a_mean * scaling
            a_min = a_mean / scaling
        else:
            # Calculate the differences in the target values
            dif = pdist(Y[:, 0:1])
            dif = asarray(dif, dtype=self.dtype)
            # Remove zero differences
            dif = dif[dif != 0.0]
            # Check that all the targets are not the same
            if len(dif) == 0:
                dif = asarray([1.0], dtype=self.dtype)
            a_max = dif.max() * self.scale
            a_min = dif.min() / self.scale
        if self.use_log:
            return asarray([[log(a_min), log(a_max)]], dtype=self.dtype)
        return asarray([[a_min, a_max]], dtype=self.dtype)

    def get_prior_mean(self, X, Y, model, **kwargs):
        "Get the prior mean value for the target only (without derivatives)."
        # Update the prior mean used in ML model
        model.prior.update(X, Y)
        return model.prior.get(X, Y[:, 0:1], get_derivatives=False)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            bounds_dict=self.bounds_dict,
            scale=self.scale,
            use_log=self.use_log,
            max_length=self.max_length,
            use_derivatives=self.use_derivatives,
            use_prior_mean=self.use_prior_mean,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
