from numpy import array, asarray, sum as sum_, sqrt
from .boundary import HPBoundaries


class UpdatingBoundaries(HPBoundaries):
    """
    An updating boundary conditions for the hyperparameters.
    Previous solutions to the hyperparameters can be used to
    updating the boundary conditions.
    The bounds and the solutions are treated as Normal distributions.
    A Normal distribution of a mixture model is then treated as
    the updated boundary conditions.
    """

    def __init__(
        self,
        bounds=None,
        sols=[],
        sol_var=0.5,
        bound_weight=4,
        min_solutions=4,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the boundary conditions for the hyperparameters.

        Parameters:
            bounds: Boundary condition class
                A Boundary condition class that make the boundaries
                of the hyperparameters.
            sols: list of dict
                The solutions of the hyperparameters from
                previous optimizations.
            sol_var: float
                The known variance of the Normal distribution used
                for the solutions.
            bound_weight: int
                The weight of the given boundary conditions in terms of
                number of solution samples.
            min_solutions: int
                The minimum number of solutions before the boundary conditions
                are updated.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set the default boundary conditions
        if bounds is None:
            bounds = HPBoundaries(
                bounds_dict={},
                use_log=True,
                seed=seed,
                dtype=dtype,
            )
        # Set all the arguments
        self.update_arguments(
            bounds=bounds,
            sols=sols,
            sol_var=sol_var,
            bound_weight=bound_weight,
            min_solutions=min_solutions,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def update_bounds(self, model, X, Y, parameters, **kwargs):
        """
        Create and update the boundary conditions for the hyperparameters.
        Therefore the variable transformation parameters are also updated.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
            X: (N,D) array
                Training features with N data points and D dimensions.
            Y: (N,1) array or (N,D+1) array
                Training targets with or without derivatives with
                N data points.
            parameters: (H) list of strings
                A list of names of the hyperparameters.

        Returns:
            self: The object itself.
        """
        # Update the parameters used
        self.make_parameters_set(parameters)
        # Update the boundary conditions and get them
        self.bounds.update_bounds(model, X, Y, parameters)
        bounds_dict = self.bounds.get_bounds(use_array=False)
        # Get length of the solution
        sol_len = len(self.sols)
        # If not enough solutions are given, then use given bounds
        if sol_len < self.min_solutions:
            self.bounds_dict = bounds_dict
            return self
        # Calculate the effective number of solutions and default boundaries
        n_eff = self.bound_weight + sol_len
        # Initialize boundary dictionary
        self.bounds_dict = {}
        for para in bounds_dict.keys():
            # Get the solutions
            sol_means = array(
                [sol["hp"][para] for sol in self.sols],
                dtype=self.dtype,
            )
            # The mean and variance from the boundary conditions
            bound_mean = sum_(bounds_dict[para], axis=-1)
            bound_var = (0.5 * (bounds_dict[para][:, 1] - bound_mean)) ** 2
            # Calculate the middle of the boundary conditions
            mean = (
                sum_(sol_means, axis=0) + (self.bound_weight * bound_mean)
            ) / n_eff
            # Calculate the variance of the solutions
            var_sols = sum_((sol_means - mean) ** 2, axis=0) + (
                self.sol_var * sol_len
            )
            # Calculate the variance of the boundary conditions
            var_bound = (self.bound_weight * ((bound_mean - mean) ** 2)) + (
                self.bound_weight * bound_var
            )
            # Calculate the distance to the boundaries from the middle
            bound_dist = 2.0 * sqrt((var_sols + var_bound) / n_eff)
            # Store the boundary conditions
            self.bounds_dict[para] = asarray(
                [mean - bound_dist, mean + bound_dist],
                dtype=self.dtype,
            ).T
        return self

    def set_dtype(self, dtype, **kwargs):
        super().set_dtype(dtype, **kwargs)
        self.bounds.set_dtype(dtype, **kwargs)
        return self

    def set_seed(self, seed=None, **kwargs):
        super().set_seed(seed, **kwargs)
        self.bounds.set_seed(seed, **kwargs)
        return self

    def update_arguments(
        self,
        bounds=None,
        sols=None,
        sol_var=None,
        bound_weight=None,
        min_solutions=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            bounds: Boundary condition class
                A Boundary condition class that make the boundaries
                of the hyperparameters.
            sols: list of dict
                The solutions of the hyperparameters from
                previous optimizations.
            sol_var: float
                The known variance of the Normal distribution used
                for the solutions.
            bound_weight: int
                The weight of the given boundary conditions in terms of
                number of solution samples.
            min_solutions: int
                The minimum number of solutions before the boundary conditions
                are updated.
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
        # Set the seed
        if seed is not None or not hasattr(self, "seed"):
            self.set_seed(seed)
        # Set the data type
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype)
        if bounds is not None:
            self.initiate_bounds_dict(bounds)
        if sols is not None:
            self.sols = [sol.copy() for sol in sols]
        if sol_var is not None:
            self.sol_var = float(sol_var)
        if bound_weight is not None:
            self.bound_weight = int(bound_weight)
        if min_solutions is not None:
            self.min_solutions = int(min_solutions)
        return self

    def initiate_bounds_dict(self, bounds, **kwargs):
        "Make and store the hyperparameter bounds."
        # Copy the boundary condition object
        self.bounds = bounds.copy()
        self.bounds_dict = self.bounds.get_bounds(use_array=False)
        # Extract the hyperparameter names
        self.parameters_set = sorted(self.bounds_dict.keys())
        self.parameters = sum(
            [
                [para] * len(self.bounds_dict[para])
                for para in self.parameters_set
            ],
            [],
        )
        # Make sure log-scale of the hyperparameters are used
        if self.bounds.use_log is False:
            raise ValueError(
                "The Updating Boundaries need to "
                "use boundary conditions in the log-scale!"
            )
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            bounds=self.bounds,
            sols=self.sols,
            sol_var=self.sol_var,
            bound_weight=self.bound_weight,
            min_solutions=self.min_solutions,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict(bounds_dict=self.bounds_dict)
        return arg_kwargs, constant_kwargs, object_kwargs
