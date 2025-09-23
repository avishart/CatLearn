from numpy import (
    array,
    asarray,
    fill_diagonal,
    full,
    inf,
    log,
    median,
    ndarray,
    sqrt,
    zeros,
)
from scipy.spatial.distance import pdist, squareform
from .boundary import HPBoundaries


class LengthBoundaries(HPBoundaries):
    """
    Boundary conditions for the hyperparameters with educated guess for
    the length-scale hyperparameter.
    Machine precisions are used as default boundary conditions for
    the rest of the hyperparameters not given in the dictionary.
    """

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
        self.update_arguments(
            bounds_dict=bounds_dict,
            scale=scale,
            use_log=use_log,
            max_length=max_length,
            use_derivatives=use_derivatives,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def set_use_derivatives(self, use_derivatives, **kwargs):
        """
        Set whether to use derivatives for the targets.

        Parameters:
            use_derivatives: bool
                Use derivatives/gradients for targets.

        Returns:
            self: The updated object itself.
        """
        self.use_derivatives = use_derivatives
        return self

    def update_arguments(
        self,
        bounds_dict=None,
        scale=None,
        use_log=None,
        max_length=None,
        use_derivatives=None,
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
                The use_derivatives will be updated when
                update_bounds is called.
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
            seed=seed,
            dtype=dtype,
        )
        # Update the parameters of the class itself
        if max_length is not None:
            self.max_length = max_length
        if use_derivatives is not None:
            self.set_use_derivatives(use_derivatives)
        return self

    def make_bounds(self, model, X, Y, parameters, parameters_set, **kwargs):
        eps_lower, eps_upper = self.get_boundary_limits()
        self.get_use_derivatives(model)
        bounds = {}
        for para in parameters_set:
            if para == "length":
                bounds[para] = self.length_bound(X, parameters.count(para))
            elif para in self.bounds_dict:
                bounds[para] = array(self.bounds_dict[para], dtype=self.dtype)
            else:
                bounds[para] = full(
                    (parameters.count(para), 2),
                    [eps_lower, eps_upper],
                    dtype=self.dtype,
                )
        return bounds

    def length_bound(self, X, l_dim, **kwargs):
        """
        Get the minimum and maximum ranges of the length-scale
        in the educated guess regime within a scale.
        """
        # Get the minimum and maximum machine precision for exponential terms
        exp_lower = sqrt(-1.0 / log(self.eps)) / self.scale
        exp_max = sqrt(-1.0 / log(1 - self.eps)) * self.scale
        # Use a smaller maximum boundary if only one length-scale is used
        if not self.max_length or l_dim == 1:
            exp_max = 2.0 * self.scale
        # Scale the convergence if derivatives of targets are used
        if self.use_derivatives:
            exp_lower = exp_lower * 0.05
        lengths = zeros((l_dim, 2), dtype=self.dtype)
        # If only one features is given then end
        if len(X) == 1:
            lengths[:, 0] = exp_lower
            lengths[:, 1] = exp_max
            if self.use_log:
                return log(lengths)
            return lengths
        # Ensure that the features are a matrix
        if not isinstance(X[0], (list, ndarray)):
            X = asarray([fp.get_vector() for fp in X], dtype=self.dtype)
        for d in range(l_dim):
            # Calculate distances
            if l_dim == 1:
                dis = pdist(X)
            else:
                d1 = d + 1
                dis = pdist(X[:, d:d1])
            dis = asarray(dis, dtype=self.dtype)
            # Calculate the maximum length-scale
            dis_max = exp_max * dis.max()
            if dis_max == 0.0:
                dis_min, dis_max = exp_lower, exp_max
            else:
                # The minimum length-scale from the nearest neighbor distance
                dis_min = exp_lower * median(self.nearest_neighbors(dis))
                if dis_min == 0.0:
                    dis_min = exp_lower
            # Transform into log-scale if specified
            lengths[d, 0], lengths[d, 1] = dis_min, dis_max
        if self.use_log:
            return log(lengths)
        return lengths

    def nearest_neighbors(self, dis, **kwargs):
        "Nearest neighbor distance."
        dis_matrix = squareform(dis)
        fill_diagonal(dis_matrix, inf)
        return dis_matrix.min(axis=1)

    def get_use_derivatives(self, model, **kwargs):
        "Get whether the derivatives of targets are used in the model."
        self.use_derivatives = model.use_derivatives
        return self.use_derivatives

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            bounds_dict=self.bounds_dict,
            scale=self.scale,
            use_log=self.use_log,
            max_length=self.max_length,
            use_derivatives=self.use_derivatives,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
