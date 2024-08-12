import numpy as np
from scipy.spatial.distance import pdist, squareform
from .boundary import HPBoundaries


class LengthBoundaries(HPBoundaries):
    def __init__(
        self,
        bounds_dict={},
        scale=1.0,
        log=True,
        max_length=True,
        use_derivatives=False,
        **kwargs,
    ):
        """
        Boundary conditions for the hyperparameters with educated guess for
        the length-scale hyperparameter.
        Machine precisions are used as default boundary conditions for
        the rest of the hyperparameters not given in the dictionary.

        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays
                with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
            max_length : bool
                Whether to use the maximum scaling for the length-scale or
                use a more reasonable scaling.
            use_derivatives : bool
                Whether the derivatives of the target are used in the model.
                The boundary conditions of the length-scale hyperparameter(s)
                will change with the use_derivatives.
                The use_derivatives will be updated when
                update_bounds is called.
        """
        self.update_arguments(
            bounds_dict=bounds_dict,
            scale=scale,
            log=log,
            max_length=max_length,
            use_derivatives=use_derivatives,
            **kwargs,
        )

    def update_arguments(
        self,
        bounds_dict=None,
        scale=None,
        log=None,
        max_length=None,
        use_derivatives=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays
                with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
            max_length : bool
                Whether to use the maximum scaling for the length-scale or
                use a more reasonable scaling.
            use_derivatives : bool
                Whether the derivatives of the target are used in the model.
                The boundary conditions of the length-scale hyperparameter(s)
                will change with the use_derivatives.
                The use_derivatives will be updated when
                update_bounds is called.

        Returns:
            self: The updated object itself.
        """
        if bounds_dict is not None:
            self.initiate_bounds_dict(bounds_dict)
        if scale is not None:
            self.scale = scale
        if log is not None:
            self.log = log
        if max_length is not None:
            self.max_length = max_length
        if use_derivatives is not None:
            self.use_derivatives = use_derivatives
        return self

    def make_bounds(self, model, X, Y, parameters, parameters_set, **kwargs):
        eps_lower, eps_upper = self.get_boundary_limits()
        self.get_use_derivatives(model)
        bounds = {}
        for para in parameters_set:
            if para == "length":
                bounds[para] = self.length_bound(X, parameters.count(para))
            elif para in self.bounds_dict:
                bounds[para] = self.bounds_dict[para].copy()
            else:
                bounds[para] = np.full(
                    (parameters.count(para), 2), [eps_lower, eps_upper]
                )
        return bounds

    def length_bound(self, X, l_dim, **kwargs):
        """
        Get the minimum and maximum ranges of the length-scale
        in the educated guess regime within a scale.
        """
        # Get the minimum and maximum machine precision for exponential terms
        exp_lower = np.sqrt(-1 / np.log(np.finfo(float).eps)) / self.scale
        exp_max = np.sqrt(-1 / np.log(1 - np.finfo(float).eps)) * self.scale
        # Use a smaller maximum boundary if only one length-scale is used
        if not self.max_length or l_dim == 1:
            exp_max = 2.0 * self.scale
        # Scale the convergence if derivatives of targets are used
        if self.use_derivatives:
            exp_lower = exp_lower * 0.05
        lengths = np.zeros((l_dim, 2))
        # If only one features is given then end
        if len(X) == 1:
            lengths[:, 0] = exp_lower
            lengths[:, 1] = exp_max
            if self.log:
                return np.log(lengths)
            return lengths
        # Ensure that the features are a matrix
        if not isinstance(X[0], (list, np.ndarray)):
            X = np.array([fp.get_vector() for fp in X])
        for d in range(l_dim):
            # Calculate distances
            if l_dim == 1:
                dis = pdist(X)
            else:
                dis = pdist(X[:, d : d + 1])
            # Calculate the maximum length-scale
            dis_max = exp_max * np.max(dis)
            if dis_max == 0.0:
                dis_min, dis_max = exp_lower, exp_max
            else:
                # The minimum length-scale from the nearest neighbor distance
                dis_min = exp_lower * np.median(self.nearest_neighbors(dis))
                if dis_min == 0.0:
                    dis_min = exp_lower
            # Transform into log-scale if specified
            lengths[d, 0], lengths[d, 1] = dis_min, dis_max
        if self.log:
            return np.log(lengths)
        return lengths

    def nearest_neighbors(self, dis, **kwargs):
        "Nearest neighbor distance."
        dis_matrix = squareform(dis)
        np.fill_diagonal(dis_matrix, np.inf)
        return np.min(dis_matrix, axis=1)

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
            log=self.log,
            max_length=self.max_length,
            use_derivatives=self.use_derivatives,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
