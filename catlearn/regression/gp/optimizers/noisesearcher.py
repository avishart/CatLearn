from numpy import nanargmin
from .linesearcher import (
    FineGridSearch,
    GoldenSearch,
    LineSearchOptimizer,
    LocalOptimizer,
    TransGridSearch,
)


class NoiseGrid(LineSearchOptimizer):
    """
    The grid method is used as the line search optimizer.
    The grid of relative-noise hyperparameter values is calculated
    with the objective function.
    The lowest of objective function values of the single grid
    is used as the optimum.
    A line of the relative-noise hyperparameter is required to
    run the line search.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=False,
        parallel=False,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the relative-noise search optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                The line search optimizers cannot use gradients
                of the objective function.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This optimizer can not be parallelized.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set all the arguments
        self.update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, line, parameters, model, X, Y, pdis, **kwargs):
        # Get the function arguments
        func_args = self.get_func_arguments(
            parameters,
            model,
            X,
            Y,
            pdis,
            self.jac,
            **kwargs,
        )
        # Calculate function values for line coordinates
        len_l = len(line)
        line = line.reshape(len_l, -1)
        f_list = self.calculate_values(line, func, func_args=func_args)
        # Find the optimal value
        i_min = nanargmin(f_list)
        sol = {
            "fun": f_list[i_min],
            "x": line[i_min],
            "success": False,
            "nfev": len_l,
            "nit": len_l,
        }
        return sol

    def set_parallel(self, parallel=False, **kwargs):
        # This optimizer can not be parallelized
        self.parallel = False
        return self

    def update_arguments(
        self,
        maxiter=None,
        jac=None,
        parallel=None,
        seed=None,
        dtype=None,
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
                The line search optimizers cannot use gradients
                of the objective function.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This optimizer can not be parallelized.
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
        super(LocalOptimizer, self).update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        return self

    def get_func_arguments(
        self,
        parameters,
        model,
        X,
        Y,
        pdis,
        jac,
        func_args=(),
        **kwargs,
    ):
        "Get the arguments needed for the objective function."
        return func_args

    def get_fun(self, func, **kwargs):
        "Get the function that evaluates the objective function."
        return func.get_eig_fun

    def calculate_values(self, thetas, func, func_args=(), **kwargs):
        "Calculate a list of values with a function."
        return func.get_all_eig_fun(thetas, *func_args)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class NoiseGoldenSearch(GoldenSearch):
    """
    The golden section search method is used as the line search optimizer.
    The line search optimizer is used for optimzing the objective function
    wrt. the relative-noise hyperparameter.
    A line of the relative-noise hyperparameter is required to
    run the line search.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=False,
        parallel=False,
        seed=None,
        dtype=float,
        tol=1e-5,
        optimize=True,
        multiple_min=False,
        theta_index=0,
        xtol=None,
        ftol=None,
        **kwargs,
    ):
        """
        Initialize the relative-noise search optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                The line search optimizers cannot use gradients
                of the objective function.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This optimizer can not be parallelized.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.
            optimize: bool
                Whether to optimize the line given by split it
                into smaller intervals.
            multiple_min: bool
                Whether to optimize multiple minimums or just
                optimize the lowest minimum.
            theta_index: int or None
                The index of the relative-noise hyperparameter that
                is optimized with the line search.
                If theta_index=None, then it will use the index of
                the relative-noise.
                If theta_index=None and no relative-noise, then theta_index=0.
            xtol: float
                A tolerance criterion of the hyperparameter for convergence.
            ftol: float
                A tolerance criterion of the objective function
                for convergence.
        """
        super().__init__(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            tol=tol,
            optimize=optimize,
            multiple_min=multiple_min,
            theta_index=theta_index,
            xtol=xtol,
            ftol=ftol,
            **kwargs,
        )

    def get_func_arguments(
        self,
        parameters,
        model,
        X,
        Y,
        pdis,
        jac,
        func_args=(),
        **kwargs,
    ):
        "Get the arguments needed for the objective function."
        return func_args

    def get_fun(self, func, **kwargs):
        "Get the function that evaluates the objective function."
        return func.get_eig_fun

    def calculate_values(self, thetas, func, func_args=(), **kwargs):
        "Calculate a list of values with a function."
        return func.get_all_eig_fun(thetas, *func_args)

    def set_jac(self, jac=False, **kwargs):
        # Line search optimizers cannot use gradients of the objective function
        self.jac = False
        return self

    def set_parallel(self, parallel=False, **kwargs):
        # This optimizer can not be parallelized
        self.parallel = False
        return self


class NoiseFineGridSearch(FineGridSearch):
    """
    The fine grid search method is used as the line search optimizer.
    The line search optimizer is used for optimzing the objective function
    wrt. the relative-noise hyperparameter.
    Finer grids are made for all minimums of the objective function.
    A line of the relative-noise hyperparameter is required to
    run the line search.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=False,
        parallel=False,
        seed=None,
        dtype=float,
        tol=1e-5,
        optimize=True,
        multiple_min=False,
        ngrid=80,
        loops=2,
        theta_index=0,
        xtol=None,
        ftol=None,
        **kwargs,
    ):
        """
        Initialize the relative-noise search optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                The line search optimizers cannot use gradients
                of the objective function.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This optimizer can not be parallelized.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.
            optimize: bool
                Whether to optimize the line given by split it
                into smaller intervals.
            multiple_min: bool
                Whether to optimize multiple minimums or just
                optimize the lowest minimum.
            theta_index: int or None
                The index of the relative-noise hyperparameter that
                is optimized with the line search.
                If theta_index=None, then it will use the index of
                the relative-noise.
                If theta_index=None and no relative-noise, then theta_index=0.
            xtol: float
                A tolerance criterion of the hyperparameter for convergence.
            ftol: float
                A tolerance criterion of the objective function
                for convergence.
        """
        super().__init__(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            tol=tol,
            optimize=optimize,
            multiple_min=multiple_min,
            ngrid=ngrid,
            loops=loops,
            theta_index=theta_index,
            xtol=xtol,
            ftol=ftol,
            **kwargs,
        )

    def get_func_arguments(
        self,
        parameters,
        model,
        X,
        Y,
        pdis,
        jac,
        func_args=(),
        **kwargs,
    ):
        "Get the arguments needed for the objective function."
        return func_args

    def get_fun(self, func, **kwargs):
        "Get the function that evaluates the objective function."
        return func.get_eig_fun

    def calculate_values(self, thetas, func, func_args=(), **kwargs):
        "Calculate a list of values with a function."
        return func.get_all_eig_fun(thetas, *func_args)

    def set_jac(self, jac=False, **kwargs):
        # Line search optimizers cannot use gradients of the objective function
        self.jac = False
        return self

    def set_parallel(self, parallel=False, **kwargs):
        # This optimizer can not be parallelized
        self.parallel = False
        return self


class NoiseTransGridSearch(TransGridSearch):
    """
    The variable transformed grid search method is used as
    the line search optimizer.
    The line search optimizer is used for optimzing the objective function
    wrt. the relative-noise hyperparameter.
    Grids are made by updating the variable transformation from
    the objective function values.
    A line of the relative-noise hyperparameter is required to
    run the line search.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=False,
        parallel=False,
        seed=None,
        dtype=float,
        tol=1e-5,
        optimize=True,
        multiple_min=False,
        ngrid=80,
        loops=2,
        use_likelihood=True,
        theta_index=0,
        xtol=None,
        ftol=None,
        **kwargs,
    ):
        """
        Initialize the relative-noise search optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                The line search optimizers cannot use gradients
                of the objective function.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This optimizer can not be parallelized.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            tol: float
                A tolerance criterion for convergence.
            optimize: bool
                Whether to optimize the line given by split it
                into smaller intervals.
            multiple_min: bool
                Whether to optimize multiple minimums or just
                optimize the lowest minimum.
            ngrid: int
                The number of grid points of the hyperparameter that
                is optimized.
            loops: int
                The number of loops where the grid points are made.
            use_likelihood: bool
                Whether to use the objective function as
                a log-likelihood or not.
                If the use_likelihood=False, the objective function is scaled
                and shifted with the maximum value.
            theta_index: int or None
                The index of the relative-noise hyperparameter that
                is optimized with the line search.
                If theta_index=None, then it will use the index of
                the relative-noise.
                If theta_index=None and no relative-noise, then theta_index=0.
            xtol: float
                A tolerance criterion of the hyperparameter for convergence.
            ftol: float
                A tolerance criterion of the objective function
                for convergence.
        """
        super().__init__(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            tol=tol,
            optimize=optimize,
            multiple_min=multiple_min,
            ngrid=ngrid,
            loops=loops,
            use_likelihood=use_likelihood,
            theta_index=theta_index,
            xtol=xtol,
            ftol=ftol,
            **kwargs,
        )

    def get_func_arguments(
        self,
        parameters,
        model,
        X,
        Y,
        pdis,
        jac,
        func_args=(),
        **kwargs,
    ):
        "Get the arguments needed for the objective function."
        return func_args

    def get_fun(self, func, **kwargs):
        "Get the function that evaluates the objective function."
        return func.get_eig_fun

    def calculate_values(self, thetas, func, func_args=(), **kwargs):
        "Calculate a list of values with a function."
        return func.get_all_eig_fun(thetas, *func_args)

    def set_jac(self, jac=False, **kwargs):
        # Line search optimizers cannot use gradients of the objective function
        self.jac = False
        return self

    def set_parallel(self, parallel=False, **kwargs):
        # This optimizer can not be parallelized
        self.parallel = False
        return self
