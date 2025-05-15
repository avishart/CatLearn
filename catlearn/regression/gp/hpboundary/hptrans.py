from numpy import (
    abs as abs_,
    array,
    concatenate,
    exp,
    finfo,
    full,
    linspace,
    log,
    where,
)
from .boundary import HPBoundaries
from .strict import StrictBoundaries


class VariableTransformation(HPBoundaries):
    def __init__(
        self,
        var_dict={},
        bounds=None,
        s=0.14,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Make variable transformation of hyperparameters into
        an interval of (0,1).
        A dictionary of mean and standard deviation values are used
        to make Logistic transformations.
        Boundary conditions can be used to calculate
        the variable transformation parameters.

        Parameters:
            var_dict: dict
                A dictionary with the variable transformation
                parameters (mean,std) for each hyperparameter.
            bounds: Boundary condition class
                A Boundary condition class that make the boundaries
                of the hyperparameters.
                The boundaries are used to calculate
                the variable transformation parameters.
            s: float
                The scale parameter in a Logistic distribution.
                It determines how large part of the distribution that
                is within the boundaries.
                s=0.5*p/(ln(p)-ln(1-p)) with p being the quantile that
                the boundaries constitute.
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
            bounds = StrictBoundaries(
                bounds_dict={},
                scale=1.0,
                use_log=True,
                use_prior_mean=True,
                seed=seed,
                dtype=dtype,
            )
        # Set all the arguments
        self.update_arguments(
            var_dict=var_dict,
            bounds=bounds,
            s=s,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def update_bounds(self, model, X, Y, parameters, **kwargs):
        """
        Create and update the boundary conditions for the hyperparameters.
        Therefore, the variable transformation parameters are also updated.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
            X: (N,D) array
                Training features with N data points and D dimensions.
            Y: (N,1) array or (N,D+1) array
                Training targets with or without derivatives
                with N data points.
            parameters: (H) list of strings
                A list of names of the hyperparameters.

        Returns:
            self: The object itself.
        """
        # Update the parameters used
        self.make_parameters_set(parameters)
        # Update the boundary conditions and get them
        self.bounds.update_bounds(model, X, Y, parameters)
        self.bounds_dict = self.bounds.get_bounds(use_array=False)
        # Update the variable transformation parameters
        for para, bound in self.bounds_dict.items():
            self.var_dict[para] = {
                "mean": bound.mean(axis=1),
                "std": self.s * abs_(bound[:, 1] - bound[:, 0]),
            }
        return self

    def get_variable_transformation_parameters(
        self,
        parameters=None,
        use_array=False,
        **kwargs,
    ):
        """
        Get the variable transformation parameters.

        Parameters:
            parameters: list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            use_array: bool
                Whether to get an array for the mean and std or
                a dictionary as output.

        Returns:
            dict: A dictionary of the variable transformation parameters.
            If use_array=True, a dictionary with mean and std is given instead.
        """
        # Make the sorted unique hyperparameters if they are given
        parameters_set = self.get_parameters_set(parameters=parameters)
        if use_array:
            var_dict_array = {}
            var_dict_array["mean"] = concatenate(
                [self.var_dict[para]["mean"] for para in parameters_set],
                axis=0,
            )
            var_dict_array["std"] = concatenate(
                [self.var_dict[para]["std"] for para in parameters_set],
                axis=0,
            )
            return var_dict_array
        return {para: self.var_dict[para].copy() for para in parameters_set}

    def transformation(self, hp, use_array=False, **kwargs):
        """
        Transform the hyperparameters with the variable transformation
        to get a dictionary.

        Parameters:
            hp: dict
                The dictionary of the hyperparameters
            use_array: bool
                Whether to get an array or a dictionary as output.

        Returns:
            (H) array: The variable transformed hyperparameters as an array
                if use_array=True.
            or
            dict: A dictionary of the variable transformed hyperparameters.
        """
        if use_array:
            return concatenate(
                [
                    self.transform(
                        theta,
                        self.var_dict[para]["mean"],
                        self.var_dict[para]["std"],
                    )
                    for para, theta in hp.items()
                ],
            )
        return {
            para: self.transform(
                theta,
                self.var_dict[para]["mean"],
                self.var_dict[para]["std"],
            )
            for para, theta in hp.items()
        }

    def reverse_trasformation(self, t, use_array=False, **kwargs):
        """
        Transform the variable transformed hyperparameters back
        to the hyperparameters dictionary.

        Parameters:
            t: dict
                The dictionary of the variable transformed hyperparameters
            use_array: bool
                Whether to get an array or a dictionary as output.

        Returns:
            (H) array: The retransformed hyperparameters as an array
                if use_array=True.
            or
            dict: A dictionary of the retransformed hyperparameters.
        """
        if use_array:
            return concatenate(
                [
                    self.retransform(
                        ti,
                        self.var_dict[para]["mean"],
                        self.var_dict[para]["std"],
                    )
                    for para, ti in t.items()
                ],
            )
        return {
            para: self.retransform(
                ti,
                self.var_dict[para]["mean"],
                self.var_dict[para]["std"],
            )
            for para, ti in t.items()
        }

    def get_bounds(
        self,
        parameters=None,
        use_array=False,
        transformed=False,
        **kwargs,
    ):
        """
        Get the boundary conditions of hyperparameters.

        Parameters :
            parameters: list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            use_array: bool
                Whether to get an array or a dictionary as output.
            transformed: bool
                If transformed=True, the boundaries is in
                variable transformed space.
                If transformed=False, the boundaries is transformed back
                to hyperparameter space.

        Returns:
            (H,2) array: The boundary conditions as an array if use_array=True.
            or
            dict: A dictionary of the boundary conditions.
        """
        # Get the bounds in the variable transformed space
        if transformed:
            if use_array:
                n_parameters = self.get_n_parameters(parameters=parameters)
                return full(
                    (n_parameters, 2),
                    [self.eps, 1.00 - self.eps],
                    dtype=self.dtype,
                )
            # Make the sorted unique hyperparameters if they are given
            parameters_set = self.get_parameters_set(parameters=parameters)
            return {
                para: full(
                    (len(self.bounds_dict[para]), 2),
                    [self.eps, 1.00 - self.eps],
                    dtype=self.dtype,
                )
                for para in parameters_set
            }
        # Get the bounds in the hyperparameter space
        return self.bounds.get_bounds(
            parameters=parameters,
            use_array=use_array,
        )

    def get_hp(
        self,
        parameters=None,
        use_array=False,
        transformed=False,
        **kwargs,
    ):
        """
        Get the guess of the hyperparameters.
        The mean of the boundary conditions in log-space is used as the guess.

        Parameters:
            parameters: list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            use_array: bool
                Whether to get an array or a dictionary as output.
            transformed: bool
                If transformed=True, the boundaries is in
                variable transformed space.
                If transformed=False, the boundaries is transformed back
                to hyperparameter space.

        Returns:
            (H) array: The guesses of the hyperparameters as an array
                if use_array=True.
            or
            dict: A dictionary of the guesses of the hyperparameters.
        """
        # Get the hyperparameter guess in the variable transformed space (0.5)
        if transformed:
            if use_array:
                n_parameters = self.get_n_parameters(parameters=parameters)
                return full((n_parameters), 0.50, dtype=self.dtype)
            # Make the sorted unique hyperparameters if they are given
            parameters_set = self.get_parameters_set(parameters=parameters)
            return {
                para: full(
                    (len(self.bounds_dict[para])),
                    0.50,
                    dtype=self.dtype,
                )
                for para in parameters_set
            }
        # Get the hyperparameter guess in the hyperparameter space
        return self.bounds.get_hp(parameters=parameters, use_array=use_array)

    def make_lines(
        self,
        parameters=None,
        ngrid=80,
        transformed=False,
        **kwargs,
    ):
        """
        Make grid in each dimension of the hyperparameters from
        the boundary conditions.

        Parameters:
            ngrid: int or (H) list
                An integer or a list with number of grid points
                in each dimension.
            transformed: bool
                If transformed=True, the grid is in variable transformed space.
                If transformed=False, the grid is transformed back
                to hyperparameter space.

        Returns:
            (H,) list: A list with grid points for each (H) hyperparameters.
        """
        # Get the number of hyperparameters
        n_parameters = self.get_n_parameters(parameters=parameters)
        # Make sure that a list of number grid points is used
        if isinstance(ngrid, (int, float)):
            ngrid = [int(ngrid)] * n_parameters
        # The grid is made within the variable transformed hyperparameters
        if transformed:
            return [
                linspace(self.eps, 1.00 - self.eps, ngrid[i], dtype=self.dtype)
                for i in range(n_parameters)
            ]
        # The grid is within the transformed space and it is then retransformed
        var_dict_array = self.get_variable_transformation_parameters(
            parameters=parameters,
            use_array=True,
        )
        lines = []
        for i, (vt_mean, vt_std) in enumerate(
            zip(
                var_dict_array["mean"],
                var_dict_array["std"],
            )
        ):
            t_line = linspace(
                self.eps,
                1.00 - self.eps,
                ngrid[i],
                dtype=self.dtype,
            )
            lines.append(self.retransform(t_line, vt_mean, vt_std))
        return lines

    def make_single_line(
        self,
        parameter,
        ngrid=80,
        i=0,
        transformed=False,
        **kwargs,
    ):
        """
        Make grid in each dimension of the hyperparameters from
        the boundary conditions.

        Parameters:
            parameters: str
                A string of the hyperparameter name.
            ngrid: int
                An integer with number of grid points in each dimension.
            i: int
                The index of the hyperparameter used
                if multiple hyperparameters of the same type exist.
            transformed: bool
                If transformed=True, the grid is in variable transformed space.
                If transformed=False, the grid is transformed back
                to hyperparameter space.

        Returns:
            (ngrid) array: A grid of ngrid points for
                the given hyperparameter.
        """
        # Make sure that a int of number grid points is used
        if not isinstance(ngrid, (int, float)):
            ngrid = ngrid[int(self.parameters.index(parameter) + i)]
        # The grid is made within the variable transformed hyperparameters
        t_line = linspace(self.eps, 1.00 - self.eps, ngrid, dtype=self.dtype)
        if transformed:
            return t_line
        # The grid is transformed back to hyperparameter space
        return self.retransform(
            t_line,
            self.var_dict[parameter]["mean"][i],
            self.var_dict[parameter]["std"][i],
        )

    def sample_thetas(
        self,
        parameters=None,
        npoints=50,
        transformed=False,
        **kwargs,
    ):
        """
        Sample hyperparameters from the transformed hyperparameter space.
        The sampled variable transformed hyperparameters are
        then transformed back to the hyperparameter space.

        Parameters:
            parameters: list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            npoints: int
                Number of points to sample.
            transformed: bool
                If transformed=True, the grid is in variable transformed space.
                If transformed=False, the grid is transformed back
                to hyperparameter space.

        Returns:
            (npoints,H) array: An array with sampled hyperparameters.
        """
        # Get the number of hyperparameters
        n_parameters = self.get_n_parameters(parameters=parameters)
        # Sample the hyperparameters from the transformed hyperparameter space
        samples = self.rng.uniform(
            low=self.eps,
            high=1.00 - self.eps,
            size=(npoints, n_parameters),
        )
        # The samples are made within the variable transformed hyperparameters
        if transformed:
            return samples
        # The samples are transformed back to hyperparameter space
        var_dict_array = self.get_variable_transformation_parameters(
            parameters=parameters,
            use_array=True,
        )
        for i, (vt_mean, vt_std) in enumerate(
            zip(var_dict_array["mean"], var_dict_array["std"])
        ):
            samples[:, i] = self.retransform(samples[:, i], vt_mean, vt_std)
        return samples

    def set_dtype(self, dtype, **kwargs):
        super().set_dtype(dtype, **kwargs)
        # Set the data type of the bounds
        self.bounds.set_dtype(dtype, **kwargs)
        # Set the data type of the variable transformation parameters
        if hasattr(self, "var_dict"):
            self.var_dict = {
                key: {
                    "mean": array(value["mean"], dtype=self.dtype),
                    "std": array(value["std"], dtype=self.dtype),
                }
                for key, value in self.var_dict.items()
            }
        return self

    def set_seed(self, seed=None, **kwargs):
        super().set_seed(seed, **kwargs)
        # Set the seed of the bounds
        self.bounds.set_seed(seed, **kwargs)
        return self

    def update_arguments(
        self,
        var_dict=None,
        bounds=None,
        s=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            var_dict: dict
                A dictionary with the variable transformation
                parameters (mean,std) for each hyperparameter.
            bounds: Boundary condition class
                A Boundary condition class that make the boundaries
                of the hyperparameters.
                The boundaries are used to calculate
                the variable transformation parameters.
            s: float
                The scale parameter in a Logistic distribution.
                It determines how large part of the distribution that
                is within the boundaries.
                s=0.5*p/(ln(p)-ln(1-p)) with p being the quantile that
                the boundaries constitute.
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
        # Set the boundary condition instance
        if bounds is not None:
            self.initiate_bounds_dict(bounds)
        # Set the seed
        if seed is not None or not hasattr(self, "seed"):
            self.set_seed(seed)
        # Set the data type
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype)
        if var_dict is not None:
            self.initiate_var_dict(var_dict)
        if s is not None:
            self.s = s
        return self

    def transform(self, theta, vt_mean, vt_std, **kwargs):
        "Transform the hyperparameters with the variable transformation."
        return 1.0 / (1.0 + exp(-(theta - vt_mean) / vt_std))

    def retransform(self, ti, vt_mean, vt_std, **kwargs):
        """
        Transform the variable transformed hyperparameters back
        to the hyperparameters.
        """
        return self.numeric_limits(vt_std * log(ti / (1.00 - ti)) + vt_mean)

    def numeric_limits(self, value, dh=None):
        """
        Replace hyperparameters if they are outside of
        the numeric limits in log-space.
        """
        if dh is None:
            dh = 0.1 * log(finfo(self.dtype).max)
        return where(-dh < value, where(value < dh, value, dh), -dh)

    def initiate_var_dict(self, var_dict, **kwargs):
        """
        Make and store the hyperparameter types and
        the dictionary with transformation parameters.
        """
        # Copy the variable transformation parameters
        self.var_dict = {
            key: {
                "mean": array(value["mean"], dtype=self.dtype),
                "std": array(value["std"], dtype=self.dtype),
            }
            for key, value in var_dict.items()
        }
        if "correction" in self.var_dict.keys():
            self.var_dict.pop("correction")
        # Extract the hyperparameters
        self.parameters_set = sorted(var_dict.keys())
        self.parameters = sum(
            [[para] * len(var_dict[para]) for para in self.parameters_set], []
        )
        return self

    def initiate_bounds_dict(self, bounds, **kwargs):
        "Make and store the hyperparameter bounds."
        # Copy the boundary condition object
        self.bounds = bounds.copy()
        self.bounds_dict = self.bounds.get_bounds(use_array=False)
        # Make sure log-scale of the hyperparameters are used
        if self.bounds.use_log is False:
            raise ValueError(
                "The Variable Transformation need to "
                "use boundary conditions in the log-scale!"
            )
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            var_dict=self.var_dict,
            bounds=self.bounds,
            s=self.s,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
