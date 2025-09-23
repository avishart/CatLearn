from .optimizer import Optimizer
from scipy.optimize import minimize


class LocalOptimizer(Optimizer):
    """
    The local optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=True,
        parallel=False,
        seed=None,
        dtype=float,
        tol=1e-3,
        **kwargs,
    ):
        """
        Initialize the local optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to use parallelization.
                This is not implemented for this method.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.
        """
        # Set all the arguments
        self.update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            tol=tol,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        raise NotImplementedError()

    def update_arguments(
        self,
        maxiter=None,
        jac=None,
        parallel=None,
        seed=None,
        dtype=None,
        tol=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to use parallelization.
                This is not implemented for this method.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.

        Returns:
            self: The updated object itself.
        """
        super().update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        if tol is not None:
            self.tol = tol
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            seed=self.seed,
            dtype=self.dtype,
            tol=self.tol,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class ScipyOptimizer(LocalOptimizer):
    """
    The local optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    This method uses the SciPy minimizers.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    """

    def __init__(
        self,
        maxiter=5000,
        jac=True,
        parallel=False,
        seed=None,
        dtype=float,
        tol=1e-8,
        method="l-bfgs-b",
        bounds=None,
        use_bounds=False,
        options={},
        opt_kwargs={},
        **kwargs,
    ):
        """
        Initialize the local optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to use parallelization.
                This is not implemented for this method.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.
            method: str
                The minimizer method used in SciPy.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                All global optimization methods are using boundary conditions.
            use_bounds: bool
                Whether to use the boundary conditions or not.
                Only some methods can use boundary conditions.
            options: dict
                Solver options used in the SciPy minimizer.
            opt_kwargs: dict
                Extra arguments used in the SciPy minimizer.
        """
        # Set boundary conditions
        self.bounds = None
        # Set options
        self.options = {}
        # Set optimization arguments
        self.opt_kwargs = {}
        # Set all the arguments
        self.update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            tol=tol,
            method=method,
            bounds=bounds,
            use_bounds=use_bounds,
            options=options,
            opt_kwargs=opt_kwargs,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Get the objective function arguments
        func_args = self.get_func_arguments(
            parameters,
            model,
            X,
            Y,
            pdis,
            self.jac,
        )
        # Get bounds or set it to default argument
        if self.use_bounds:
            bounds = self.make_bounds(parameters, use_array=True)
        else:
            bounds = None
        # Minimize objective function with SciPy
        sol = minimize(
            self.get_fun(func),
            x0=theta,
            method=self.method,
            jac=self.jac,
            tol=self.tol,
            args=func_args,
            bounds=bounds,
            options=self.options,
            **self.opt_kwargs,
        )
        return self.get_final_solution(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
        )

    def set_dtype(self, dtype, **kwargs):
        super().set_dtype(dtype=dtype, **kwargs)
        # Set the data type of the bounds
        if self.bounds is not None and hasattr(self.bounds, "set_dtype"):
            self.bounds.set_dtype(dtype=dtype, **kwargs)
        return self

    def set_seed(self, seed=None, **kwargs):
        super().set_seed(seed=seed, **kwargs)
        # Set the random seed of the bounds
        if self.bounds is not None and hasattr(self.bounds, "set_seed"):
            self.bounds.set_seed(seed=seed, **kwargs)
        return self

    def set_maxiter(self, maxiter, **kwargs):
        super().set_maxiter(maxiter, **kwargs)
        # Set the maximum number of iterations in the options
        if self.method in ["nelder-mead"]:
            self.options["maxfev"] = self.maxiter
        elif self.method in ["l-bfgs-b", "tnc"]:
            self.options["maxfun"] = self.maxiter
        else:
            self.options["maxiter"] = self.maxiter
        return self

    def update_arguments(
        self,
        maxiter=None,
        jac=None,
        parallel=None,
        seed=None,
        dtype=None,
        tol=None,
        method=None,
        bounds=None,
        use_bounds=None,
        options=None,
        opt_kwargs=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to use parallelization.
                This is not implemented for this method.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.
            method: str
                The minimizer method used in SciPy.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                All global optimization methods are using boundary conditions.
            use_bounds: bool
                Whether to use the boundary conditions or not.
                Only some methods can use boundary conditions.
            options: dict
                Solver options used in the SciPy minimizer.
            opt_kwargs: dict
                Extra arguments used in the SciPy minimizer.

        Returns:
            self: The updated object itself.
        """
        if method is not None:
            self.method = method.lower()
            # If method is updated then maxiter must be updated
            if maxiter is None and hasattr(self, "maxiter"):
                maxiter = self.maxiter
        if options is not None:
            self.options.update(options)
        if bounds is not None:
            self.bounds = bounds.copy()
        if use_bounds is not None:
            if self.bounds is not None and self.method in [
                "nelder-mead",
                "l-bfgs-b",
                "tnc",
                "slsqp",
                "powell",
                "trust-constr",
                "cobyla",
            ]:
                self.use_bounds = use_bounds
            else:
                self.use_bounds = False
        if opt_kwargs is not None:
            self.opt_kwargs.update(opt_kwargs)
        # Set the arguments for the parent class
        super().update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            tol=tol,
        )
        return self

    def make_bounds(self, parameters, use_array=True, **kwargs):
        "Make the boundary conditions of the hyperparameters."
        return self.bounds.get_bounds(
            parameters=parameters,
            use_array=use_array,
            **kwargs,
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            seed=self.seed,
            dtype=self.dtype,
            tol=self.tol,
            method=self.method,
            bounds=self.bounds,
            use_bounds=self.use_bounds,
            options=self.options,
            opt_kwargs=self.opt_kwargs,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class ScipyPriorOptimizer(ScipyOptimizer):
    """
    The local optimizer used for optimzing the objective function
    wrt.the hyperparameters.
    This method uses the SciPy minimizers.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    If prior distributions of the hyperparameters are used,
    it will start by include
    the prior distributions and then restart with
    excluded prior distributions.
    """

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Get solution with the prior distributions
        sol = super().run(func, theta, parameters, model, X, Y, pdis, **kwargs)
        # Check if prior distributions of the hyperparameters are used
        if pdis is None:
            return sol
        # Save the number of evaluations and the new best hyperparameters
        nfev = sol["nfev"]
        # Reset the solution in objective function instance
        self.reset_func(func)
        # Exclude the prior distributions in the optimization
        sol = super().run(
            func,
            sol["x"],
            parameters,
            model,
            X,
            Y,
            None,
            **kwargs,
        )
        sol["nfev"] += nfev
        sol["nit"] = 2
        return sol


class ScipyGuessOptimizer(ScipyOptimizer):
    """
    The local optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    This method uses the SciPy minimizers.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    Use boundary conditions to give an extra guess of the hyperparameters
    that also are optimized.
    """

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Optimize the initial hyperparameters
        sol = super().run(func, theta, parameters, model, X, Y, pdis, **kwargs)
        # Check if boundary conditions of the hyperparameters are used
        if self.bounds is None:
            return sol
        # Use the boundaries to give an educated guess of the hyperparmeters
        theta_guess = self.guess_hp(parameters, use_array=True)
        sol_ed = super().run(
            func,
            theta_guess,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )
        # Update the solution if it is better
        sol = self.compare_solutions(sol, sol_ed)
        sol["nit"] = 2
        return sol

    def guess_hp(self, parameters, use_array=True, **kwargs):
        "Make a guess of the hyperparameters from the boundary conditions."
        return self.bounds.get_hp(
            parameters=parameters,
            use_array=use_array,
            **kwargs,
        )
