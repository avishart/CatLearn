from numpy import array, asarray, finfo, full, log, sqrt
from .length import LengthBoundaries


class RestrictedBoundaries(LengthBoundaries):
    def __init__(
        self,
        bounds_dict={},
        scale=1.0,
        use_log=True,
        max_length=True,
        use_derivatives=False,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Boundary conditions for the hyperparameters with educated guess for
        the length-scale and relative-noise hyperparameters.
        Machine precisions are used as default boundary conditions for
        the rest of the hyperparameters not given in the dictionary.

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
                The use_derivatives will be updated when
                update_bounds is called.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            bounds_dict=bounds_dict,
            scale=scale,
            use_log=use_log,
            max_length=max_length,
            use_derivatives=use_derivatives,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

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
            elif para in self.bounds_dict:
                bounds[para] = array(self.bounds_dict[para], dtype=self.dtype)
            else:
                bounds[para] = full(
                    (parameters.count(para), 2),
                    [eps_lower, eps_upper],
                    dtype=self.dtype,
                )
        return bounds

    def noise_bound(
        self,
        Y,
        eps_lower=None,
        **kwargs,
    ):
        """
        Get the minimum and maximum ranges of the noise in
        the educated guess regime within a scale.
        """
        if eps_lower is None:
            eps_lower = 10.0 * sqrt(2.0 * finfo(self.dtype).eps)
        n_max = len(Y.reshape(-1)) * self.scale
        if self.use_log:
            return asarray([[eps_lower, log(n_max)]], dtype=self.dtype)
        return asarray([[eps_lower, n_max]], dtype=self.dtype)
