from numpy import (
    append,
    array,
    asarray,
    concatenate,
    nanargmin,
    ndarray,
    sort,
    sum as sum_,
    tile,
    unique,
    where,
)
from scipy import __version__ as scipy_version
from scipy.optimize import basinhopping, dual_annealing
from ase.parallel import world
from .optimizer import Optimizer
from .linesearcher import GoldenSearch
from .localoptimizer import ScipyOptimizer
from ..hpboundary import EducatedBoundaries, VariableTransformation


class GlobalOptimizer(Optimizer):
    """
    The global optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    The global optimizer requires a local optimization method and
    boundary conditions of the hyperparameters.
    """

    def __init__(
        self,
        local_optimizer=None,
        bounds=VariableTransformation(),
        maxiter=5000,
        jac=False,
        parallel=False,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default local optimizer
        if local_optimizer is None:
            local_optimizer = ScipyOptimizer(
                maxiter=maxiter,
                bounds=bounds,
                use_bounds=False,
            )
        # Set all the arguments
        self.update_arguments(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def set_dtype(self, dtype, **kwargs):
        # Set the dtype for the global optimizer
        super().set_dtype(dtype=dtype, **kwargs)
        # Set the data type of the bounds
        if self.bounds is not None and hasattr(self.bounds, "set_dtype"):
            self.bounds.set_dtype(dtype=dtype, **kwargs)
        # Set the dtype for the local optimizer
        if self.local_optimizer is not None and hasattr(
            self.local_optimizer, "set_dtype"
        ):
            self.local_optimizer.set_dtype(dtype=dtype)
        return self

    def set_seed(self, seed=None, **kwargs):
        # Set the seed for the global optimizer
        super().set_seed(seed=seed, **kwargs)
        # Set the random seed of the bounds
        if self.bounds is not None and hasattr(self.bounds, "set_seed"):
            self.bounds.set_seed(seed=seed, **kwargs)
        # Set the seed for the local optimizer
        if self.local_optimizer is not None and hasattr(
            self.local_optimizer,
            "set_seed",
        ):
            self.local_optimizer.set_seed(seed=seed, **kwargs)
        return self

    def set_maxiter(self, maxiter, **kwargs):
        super().set_maxiter(maxiter, **kwargs)
        # Set the maxiter for the local optimizer
        if self.local_optimizer is not None:
            self.local_optimizer.update_arguments(maxiter=maxiter)
        return self

    def set_jac(self, jac=True, **kwargs):
        # The gradients of the function are unused by the global optimizer
        self.jac = False
        return self

    def set_parallel(self, parallel=False, **kwargs):
        # This global optimizer can not be parallelized
        self.parallel = False
        return self

    def update_arguments(
        self,
        local_optimizer=None,
        bounds=None,
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
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
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
        # Set the local optimizer
        if local_optimizer is not None:
            self.local_optimizer = local_optimizer.copy()
        elif not hasattr(self, "local_optimizer"):
            self.local_optimizer = None
        # Set the bounds
        if bounds is not None:
            self.bounds = bounds.copy()
            # Use the same boundary conditions in the local optimizer
            self.local_optimizer.update_arguments(bounds=self.bounds)
        elif not hasattr(self, "bounds"):
            self.bounds = None
        # Set the arguments for the parent class
        super().update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        return self

    def run_local_opt(
        self,
        func,
        theta,
        parameters,
        model,
        X,
        Y,
        pdis,
        **kwargs,
    ):
        "Run the local optimization."
        return self.local_optimizer.run(
            func,
            theta,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )

    def make_lines(self, parameters, ngrid, **kwargs):
        "Make the lines of the hyperparameters from boundary conditions."
        return self.bounds.make_lines(
            parameters=parameters,
            ngrid=ngrid,
            **kwargs,
        )

    def make_bounds(self, parameters, use_array=True, **kwargs):
        "Make the boundary conditions of the hyperparameters."
        return self.bounds.get_bounds(
            parameters=parameters,
            use_array=use_array,
            **kwargs,
        )

    def sample_thetas(self, parameters, npoints, **kwargs):
        "Draw random hyperparameter samples from the boundary conditions."
        return self.bounds.sample_thetas(
            parameters=parameters,
            npoints=npoints,
            **kwargs,
        )

    def get_optimal_npoints(self, npoints, size, **kwargs):
        "Ensure that the optimal number of points is used for the size."
        npoints = int(int(npoints / size) * size)
        if npoints == 0:
            npoints = int(size)
        return npoints

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            local_optimizer=self.local_optimizer,
            bounds=self.bounds,
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


class RandomSamplingOptimizer(GlobalOptimizer):
    """
    The random sampling optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    The random sampling optimizer samples the hyperparameters randomly
    from the boundary conditions
    and optimize all samples with the local optimizer.
    """

    def __init__(
        self,
        local_optimizer=None,
        bounds=VariableTransformation(),
        maxiter=5000,
        jac=False,
        parallel=False,
        npoints=40,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            npoints: int
                The number of hyperparameter points samled from
                the boundary conditions.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default local optimizer
        if local_optimizer is None:
            local_optimizer = ScipyOptimizer(
                maxiter=int(maxiter / npoints),
                bounds=bounds,
                use_bounds=False,
            )
        # Set all the arguments
        self.update_arguments(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            npoints=npoints,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Draw random hyperparameter samples
        thetas = array([theta], dtype=self.dtype)
        if self.npoints > 1:
            thetas = self.sample_thetas(
                parameters,
                npoints=int(self.npoints - 1),
            )
            thetas = append(thetas, thetas, axis=0)
        # Make empty solution and lists
        sol = self.get_empty_solution()
        # Perform the local optimization for random samples
        sol = self.optimize_samples(
            sol,
            func,
            thetas,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )
        return sol

    def set_parallel(self, parallel=False, **kwargs):
        self.parallel = parallel
        return self

    def update_arguments(
        self,
        local_optimizer=None,
        bounds=None,
        maxiter=None,
        jac=None,
        parallel=None,
        npoints=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            npoints: int
                The number of hyperparameter points samled from
                the boundary conditions.
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
        # Set the arguments for the parent class
        super().update_arguments(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        # Set the number of points
        if npoints is not None:
            if self.parallel:
                self.npoints = self.get_optimal_npoints(npoints, world.size)
            else:
                self.npoints = int(npoints)
        return self

    def optimize_samples(
        self,
        sol,
        func,
        thetas,
        parameters,
        model,
        X,
        Y,
        pdis,
        **kwargs,
    ):
        "Perform the local optimization of the random samples."
        # Check if the optimization should be performed in parallel
        if self.parallel:
            return self.optimize_samples_parallel(
                sol,
                func,
                thetas,
                parameters,
                model,
                X,
                Y,
                pdis,
                **kwargs,
            )
        for theta in thetas:
            # Check if the maximum number of iterations is used
            if sol["nfev"] >= self.maxiter:
                break
            # Do local optimization
            sol_s = self.run_local_opt(
                func,
                theta,
                parameters,
                model,
                X,
                Y,
                pdis,
                **kwargs,
            )
            # Update the solution if it is better
            sol = self.compare_solutions(sol, sol_s)
        # Update the total number of iterations
        sol["nit"] = len(thetas)
        # Get the all best time best solution
        return self.get_final_solution(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
        )

    def optimize_samples_parallel(
        self,
        sol,
        func,
        thetas,
        parameters,
        model,
        X,
        Y,
        pdis,
        **kwargs,
    ):
        "Perform the local optimization of the random samples in parallel."
        rank, size = world.rank, world.size
        for t, theta in enumerate(thetas):
            if rank == t % size:
                # Check if the maximum number of iterations is used
                if sol["nfev"] >= self.maxiter:
                    break
                # Do local optimization in parallel
                sol_s = self.run_local_opt(
                    func,
                    theta,
                    parameters,
                    model,
                    X,
                    Y,
                    pdis,
                    **kwargs,
                )
                # Update the solution if it is better
                sol = self.compare_solutions(sol, sol_s)
        # Update the total number of iterations
        sol["nit"] = len(thetas)
        # Get the all best time best solution for all CPUs and broadcast it
        return self.get_final_solution(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            local_optimizer=self.local_optimizer,
            bounds=self.bounds,
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            npoints=self.npoints,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class GridOptimizer(GlobalOptimizer):
    """
    The grid optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    The grid optimizer makes a grid in the hyperparameter space from
    the boundary conditions and evaluate them.
    The grid point with the lowest function value can be optimized
    with the local optimizer.
    """

    def __init__(
        self,
        local_optimizer=None,
        bounds=VariableTransformation(),
        maxiter=5000,
        jac=False,
        parallel=False,
        n_each_dim=None,
        optimize=True,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            n_each_dim: int or (H) list
                An integer or a list with number of grid points
                in each dimension of the hyperparameters.
            optimize: bool
                Whether to perform a local optimization on the best
                found solution.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default local optimizer
        if local_optimizer is None:
            local_optimizer = ScipyOptimizer(
                maxiter=maxiter,
                bounds=bounds,
                use_bounds=False,
            )
        # Set all the arguments
        self.update_arguments(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            n_each_dim=n_each_dim,
            optimize=optimize,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Number of points per dimension
        n_each_dim = self.get_n_each_dim(len(theta))
        # Make grid either with the same or different numbers in each dimension
        lines = self.make_lines(parameters, ngrid=n_each_dim)
        thetas = append(
            [theta],
            self.make_grid(lines, maxiter=int(self.maxiter - 1)),
            axis=0,
        )
        # Check if the number of points is well parallized if it is used
        thetas = self.check_npoints(thetas)
        # Make empty solution and lists
        sol = self.get_empty_solution()
        # Get the function arguments
        func_args = self.get_func_arguments(
            parameters,
            model,
            X,
            Y,
            pdis,
            jac=False,
            **kwargs,
        )
        # Calculate the grid points
        f_list = self.calculate_values(thetas, func, func_args=func_args)
        # Find the minimum function value
        sol = self.get_minimum(sol, thetas, f_list)
        sol = self.get_final_solution(sol, func, parameters, model, X, Y, pdis)
        # Perform the local optimization for the minimum function value
        sol = self.optimize_minimum(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )
        return sol

    def set_parallel(self, parallel=False, **kwargs):
        self.parallel = parallel
        return self

    def update_arguments(
        self,
        local_optimizer=None,
        bounds=None,
        maxiter=None,
        jac=None,
        parallel=None,
        n_each_dim=None,
        optimize=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            n_each_dim: int or (H) list
                An integer or a list with number of grid points
                in each dimension of the hyperparameters.
            optimize: bool
                Whether to perform a local optimization on the best
                found solution.
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
        if n_each_dim is not None:
            if isinstance(n_each_dim, (list, ndarray)):
                self.n_each_dim = n_each_dim.copy()
            else:
                self.n_each_dim = n_each_dim
        elif not hasattr(self, "n_each_dim"):
            self.n_each_dim = None
        if optimize is not None:
            self.optimize = optimize
        # Set the arguments for the parent class
        super().update_arguments(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        return self

    def make_grid(self, lines, maxiter=5000):
        """
        Make a grid in multi-dimensions from a list of 1D grids
        in each dimension.
        """
        lines = array(lines, dtype=self.dtype)
        if len(lines.shape) < 2:
            lines = lines.reshape(1, -1)
        # Number of combinations
        combi = 1
        for i in [len(line) for line in lines]:
            combi *= i
        if combi < maxiter:
            maxiter = combi
        # If there is a low probability to find grid points randomly
        # the entire grid are calculated
        if (1 - (maxiter / combi)) < 0.99:
            X = lines[0].reshape(-1, 1)
            lines = lines[1:]
            for line in lines:
                dim_X = len(X)
                X = tile(X, (len(line), 1))
                X = concatenate(
                    [
                        X,
                        sort(
                            concatenate([line.reshape(-1)] * dim_X, axis=0)
                        ).reshape(-1, 1),
                    ],
                    axis=1,
                )
            return self.rng.permutation(X)[:maxiter]
        # Randomly sample the grid points
        X = asarray(
            [self.rng.choice(line, size=maxiter) for line in lines],
            dtype=self.dtype,
        ).T
        X = unique(X, axis=0)
        while len(X) < maxiter:
            x = asarray(
                [self.rng.choice(line, size=1) for line in lines],
                dtype=self.dtype,
            ).T
            X = append(X, x, axis=0)
            X = unique(X, axis=0)
        return X[:maxiter]

    def optimize_minimum(
        self,
        sol,
        func,
        parameters,
        model,
        X,
        Y,
        pdis,
        **kwargs,
    ):
        "Perform the local optimization of the found minimum."
        # Check if optimization should be used
        if not self.optimize:
            return sol
        # Check if all iterations have been used
        if sol["nfev"] >= self.maxiter:
            return sol
        # Perform local optimization
        sol_s = self.run_local_opt(
            func,
            sol["x"],
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )
        # Update the solution if it is better
        sol = self.compare_solutions(sol, sol_s)
        # Update the number of used iterations
        sol["nit"] += 1
        return self.get_final_solution(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
        )

    def get_n_each_dim(self, dim, **kwargs):
        "Number of points per dimension."
        if self.n_each_dim is None:
            n_each_dim = int((self.maxiter - 1) ** (1 / dim))
            n_each_dim = n_each_dim if n_each_dim > 1 else 1
        else:
            n_each_dim = self.n_each_dim
        return n_each_dim

    def check_npoints(self, thetas, **kwargs):
        "Check if the number of points is well parallized if it is used."
        if self.parallel:
            npoints = self.get_optimal_npoints(len(thetas), world.size)
            return thetas[:npoints]
        return thetas

    def get_minimum(self, sol, thetas, f_list, **kwargs):
        "Find the minimum function value and update the solution."
        # Find the minimum function value
        i_min = nanargmin(f_list)
        # Get the number of used iterations
        thetas_len = len(thetas)
        # Update the number of used iterations
        sol["nfev"] += thetas_len
        sol["nit"] += thetas_len
        # Check if a better point is found
        if f_list[i_min] > sol["fun"]:
            return sol
        # Update the solution if a better point is found
        sol["fun"] = f_list[i_min]
        sol["x"] = thetas[i_min].copy()
        sol["message"] = "Lower function value found."
        return sol

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            local_optimizer=self.local_optimizer,
            bounds=self.bounds,
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            n_each_dim=self.n_each_dim,
            optimize=self.optimize,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class IterativeLineOptimizer(GridOptimizer):
    """
    The iteratively line optimizer used for optimzing
    the objective function wrt. the hyperparameters.
    The iteratively line optimizer makes a 1D grid in each dimension
    of the hyperparameter space from the boundary conditions.
    The grid points are then evaluated and the best value
    updates the hyperparameter in the specific dimension.
    This process is done iteratively over all dimensions and in loops.
    The grid point with the lowest function value can be optimized
    with the local optimizer.
    """

    def __init__(
        self,
        local_optimizer=None,
        bounds=VariableTransformation(),
        maxiter=5000,
        jac=False,
        parallel=False,
        n_each_dim=None,
        loops=3,
        calculate_init=False,
        optimize=True,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            n_each_dim: int or (H) list
                An integer or a list with number of grid points
                in each dimension of the hyperparameters.
            loops: int
                The number of times all the hyperparameter dimensions
                have been searched.
            calculate_init: bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point.
            optimize: bool
                Whether to perform a local optimization on the best
                found solution.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            n_each_dim=n_each_dim,
            loops=loops,
            calculate_init=calculate_init,
            optimize=optimize,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Number of points per dimension
        n_each_dim = self.get_n_each_dim(len(theta))
        # Make grid either with the same or different numbers in each dimension
        lines = self.make_lines(parameters, ngrid=n_each_dim)
        # Get the function arguments
        func_args = self.get_func_arguments(
            parameters,
            model,
            X,
            Y,
            pdis,
            jac=False,
            **kwargs,
        )
        # Calculate the grid points in the iterative grid/line search
        sol = self.iterative_line(theta, lines, func, func_args=func_args)
        sol = self.get_final_solution(sol, func, parameters, model, X, Y, pdis)
        # Perform the local optimization for the minimum function value
        sol = self.optimize_minimum(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )
        return sol

    def update_arguments(
        self,
        local_optimizer=None,
        bounds=None,
        maxiter=None,
        jac=None,
        parallel=None,
        n_each_dim=None,
        loops=None,
        calculate_init=None,
        optimize=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            local_optimizer: Local optimizer class
                A local optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            n_each_dim: int or (H) list
                An integer or a list with number of grid points
                in each dimension of the hyperparameters.
            loops: int
                The number of times all the hyperparameter dimensions
                have been searched.
            calculate_init: bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point.
            optimize: bool
                Whether to perform a local optimization on the best
                found solution.
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
        # Set the arguments for the parent class
        super().update_arguments(
            local_optimizer=local_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            n_each_dim=None,
            optimize=optimize,
            seed=seed,
            dtype=dtype,
        )
        if loops is not None:
            self.loops = int(loops)
        if calculate_init is not None:
            self.calculate_init = calculate_init
        if n_each_dim is not None:
            if isinstance(n_each_dim, (list, ndarray)):
                if sum_(n_each_dim) * self.loops > self.maxiter:
                    self.n_each_dim = self.get_n_each_dim(len(n_each_dim))
                else:
                    self.n_each_dim = n_each_dim.copy()
            else:
                self.n_each_dim = n_each_dim
        return self

    def iterative_line(self, theta, lines, func, func_args=(), **kwargs):
        "Perform iteratively grid/line search."
        # Make an initial solution
        if self.calculate_init:
            sol = self.get_initial_solution(theta, func, func_args=func_args)
        else:
            sol = self.get_empty_solution()
        # Get the dimension list
        dims = list(range(len(lines)))
        # Set initidal dimension
        d = None
        # Perform loops
        for i in range(self.loops):
            # Permute the dimensions
            dim_perm = self.rng.permutation(dims)
            # Make sure the same dimension is not used after each other
            if dim_perm[0] == d:
                dim_perm = dim_perm[1:]
            for d in dim_perm:
                # Make the hyperparameter changes to the specific dimension
                thetas = tile(theta, (len(lines[d]), 1))
                thetas[:, d] = lines[d].copy()
                f_list = self.calculate_values(
                    thetas,
                    func,
                    func_args=func_args,
                )
                sol = self.get_minimum(sol, thetas, f_list)
                theta = sol["x"].copy()
        return sol

    def get_n_each_dim(self, dim):
        "Number of points per dimension."
        if self.n_each_dim is None:
            n_each_dim = int(self.maxiter / (self.loops * dim))
            n_each_dim = n_each_dim if n_each_dim > 1 else 1
        else:
            n_each_dim = self.n_each_dim
        if self.parallel:
            return self.get_n_each_dim_parallel(n_each_dim)
        return n_each_dim

    def get_n_each_dim_parallel(self, n_each_dim):
        "Number of points per dimension if it is parallelized."
        if isinstance(n_each_dim, (list, ndarray)):
            for d, n_dim in enumerate(n_each_dim):
                n_each_dim[d] = self.get_optimal_npoints(n_dim, world.size)
        else:
            n_each_dim = self.get_optimal_npoints(n_each_dim, world.size)
        return n_each_dim

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            local_optimizer=self.local_optimizer,
            bounds=self.bounds,
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            n_each_dim=self.n_each_dim,
            loops=self.loops,
            calculate_init=self.calculate_init,
            optimize=self.optimize,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class FactorizedOptimizer(GlobalOptimizer):
    """
    The factorized optimizer used for optimzing
    the objective function wrt. the hyperparameters.
    The factorized optimizer makes a 1D grid for each
    hyperparameter from the boundary conditions.
    The hyperparameters are then optimized with a line search optimizer.
    The line search optimizer optimizes only one of the hyperparameters and
    it therefore relies on a factorization method as
    the objective function.
    """

    def __init__(
        self,
        line_optimizer=None,
        bounds=VariableTransformation(),
        maxiter=5000,
        jac=False,
        parallel=False,
        ngrid=80,
        calculate_init=False,
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            line_optimizer: Line search optimizer class
                A line search optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            ngrid: int
                The number of grid points of the hyperparameter
                that is optimized.
            calculate_init: bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # The gradients of the function are unused by the global optimizer
        self.jac = False
        # Set default line optimizer
        if line_optimizer is None:
            line_optimizer = GoldenSearch(
                maxiter=int(maxiter),
                parallel=parallel,
            )
        # Set all the arguments
        self.update_arguments(
            line_optimizer=line_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            ngrid=ngrid,
            calculate_init=calculate_init,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Make an initial solution or use an empty solution
        if self.calculate_init:
            func_args = self.get_func_arguments(
                parameters,
                model,
                X,
                Y,
                pdis,
                jac=False,
                **kwargs,
            )
            sol = self.get_initial_solution(theta, func, func_args=func_args)
        else:
            sol = self.get_empty_solution()
        # Make the lines of the hyperparameters
        lines = asarray(self.make_lines(parameters, ngrid=self.ngrid)).T
        # Optimize the hyperparameters with the line search
        sol_s = self.run_line_opt(
            func,
            lines,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )
        # Update the solution if it is better
        sol = self.compare_solutions(sol, sol_s)
        # Change the solution message
        if sol["success"]:
            sol["message"] = "Local optimization is converged."
        else:
            sol["message"] = "Local optimization is not converged."
        return self.get_final_solution(
            sol,
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
        )

    def set_parallel(self, parallel=False, **kwargs):
        self.parallel = parallel
        return self

    def update_arguments(
        self,
        line_optimizer=None,
        bounds=None,
        maxiter=None,
        jac=None,
        parallel=None,
        ngrid=None,
        calculate_init=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            line_optimizer: Line search optimizer class
                A line search optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
                This is not implemented for this method.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
            ngrid: int
                The number of grid points of the hyperparameter
                that is optimized.
            calculate_init: bool
                Whether to calculate the initial given hyperparameters.
                If it is parallelized, all CPUs will calculate this point.
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
        # Set the arguments for the parent class
        super().update_arguments(
            local_optimizer=line_optimizer,
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        # Set the arguments
        if ngrid is not None:
            if self.parallel:
                self.ngrid = self.get_optimal_npoints(ngrid, world.size)
            else:
                self.ngrid = int(ngrid)
        if calculate_init is not None:
            self.calculate_init = calculate_init
        return self

    def run_line_opt(
        self,
        func,
        lines,
        parameters,
        model,
        X,
        Y,
        pdis,
        **kwargs,
    ):
        "Run the line search optimization."
        return self.local_optimizer.run(
            func,
            lines,
            parameters,
            model,
            X,
            Y,
            pdis,
            **kwargs,
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            line_optimizer=self.local_optimizer,
            bounds=self.bounds,
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            ngrid=self.ngrid,
            calculate_init=self.calculate_init,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class ScipyGlobalOptimizer(Optimizer):
    """
    The global optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    The global optimizer requires a local optimization method and
    boundary conditions of the hyperparameters.
    This global optimizer is a wrapper to SciPy's global optimizers.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=True,
        parallel=False,
        opt_kwargs={},
        local_kwargs={},
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
           jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's optimizer.
            local_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's local minimizer.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default arguments for SciPy's global optimizer
        self.opt_kwargs = dict()
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs = dict(options={})
        # Set all the arguments
        self.update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def set_seed(self, seed=None, **kwargs):
        super().set_seed(seed=seed, **kwargs)
        # Set the random number generator for the global optimizer
        if scipy_version >= "1.15":
            self.opt_kwargs["rng"] = self.rng
        else:
            self.opt_kwargs["seed"] = self.seed
        return self

    def set_jac(self, jac=True, **kwargs):
        self.jac = jac
        return self

    def set_parallel(self, parallel=False, **kwargs):
        # This global optimizer can not be parallelized
        self.parallel = False
        return self

    def update_arguments(
        self,
        maxiter=None,
        jac=None,
        parallel=None,
        opt_kwargs=None,
        local_kwargs=None,
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
                the global optimizer can use.
           jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's optimizer.
            local_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's local minimizer.
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
        # Set the arguments for the parent class
        super().update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            seed=seed,
            dtype=dtype,
        )
        if opt_kwargs is not None:
            self.opt_kwargs.update(opt_kwargs)
        if local_kwargs is not None:
            if "options" in local_kwargs:
                local_no_options = {
                    key: value
                    for key, value in local_kwargs.items()
                    if key != "options"
                }
                self.local_kwargs.update(local_no_options)
                self.local_kwargs["options"].update(local_kwargs["options"])
            else:
                self.local_kwargs.update(local_kwargs)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            opt_kwargs=self.opt_kwargs,
            local_kwargs=self.local_kwargs,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class BasinOptimizer(ScipyGlobalOptimizer):
    """
    The basin-hopping optimizer used for optimzing the objective function
    wrt. the hyperparameters.
    The basin-hopping optimizer is a wrapper to SciPy's basinhopping.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
    No local optimizer and boundary conditions are given to this optimizer.
    The local optimizer is set by keywords in the local_kwargs and
    it uses SciPy's minimizer.
    """

    def __init__(
        self,
        maxiter=5000,
        jac=True,
        parallel=False,
        opt_kwargs={},
        local_kwargs={},
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's basinhopping.
            local_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's local minimizer.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default arguments for SciPy's basinhopping
        self.opt_kwargs = dict(
            niter=5,
            interval=10,
            T=1.0,
            stepsize=0.1,
            niter_success=None,
        )
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs = dict(
            options={"maxiter": int(maxiter / self.opt_kwargs["niter"])}
        )
        # Set all the arguments
        self.update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
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
        # Get the function that evaluate the objective function
        fun = self.get_fun(func)
        # Set the minimizer kwargs
        minimizer_kwargs = dict(
            args=func_args,
            jac=self.jac,
            **self.local_kwargs,
        )
        # Do the basin-hopping
        sol = basinhopping(
            fun,
            x0=theta,
            minimizer_kwargs=minimizer_kwargs,
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

    def update_arguments(
        self,
        maxiter=None,
        jac=None,
        parallel=None,
        opt_kwargs=None,
        local_kwargs=None,
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
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's basinhopping.
            local_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's local minimizer.
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
        # Set the arguments for the parent class
        super().update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
        )
        # Make sure not to many iterations are used in average
        maxiter_niter = int(self.maxiter / self.opt_kwargs["niter"])
        if maxiter_niter < self.local_kwargs["options"]["maxiter"]:
            self.local_kwargs["options"]["maxiter"] = maxiter_niter
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            opt_kwargs=self.opt_kwargs,
            local_kwargs=self.local_kwargs,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AnneallingOptimizer(ScipyGlobalOptimizer):
    """
    The simulated annealing optimizer used for optimzing
    the objective function wrt. the hyperparameters.
    The simulated annealing optimizer is a wrapper to
    SciPy's dual_annealing.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
    No local optimizer is given to this optimizer.
    The local optimizer is set by keywords in the local_kwargs and
    it uses SciPy's minimizer.
    """

    def __init__(
        self,
        bounds=EducatedBoundaries(use_log=True),
        maxiter=5000,
        jac=True,
        parallel=False,
        opt_kwargs={},
        local_kwargs={},
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's dual_annealing.
            local_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's local minimizer.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default arguments for SciPy's dual_annealing
        self.opt_kwargs = dict(
            initial_temp=5230.0,
            restart_temp_ratio=2e-05,
            visit=2.62,
            accept=-5.0,
            no_local_search=False,
        )
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs = dict(options={})
        # Set all the arguments
        self.update_arguments(
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Get the function arguments
        func_args = self.get_func_arguments(
            parameters,
            model,
            X,
            Y,
            pdis,
            jac=False,
            **kwargs,
        )
        # Get the function
        fun = self.get_fun(func)
        # Set the minimizer kwargs
        minimizer_kwargs = dict(jac=False, **self.local_kwargs)
        # Make boundary conditions
        bounds = self.make_bounds(parameters, use_array=True)
        # Do the dual simulated annealing
        sol = dual_annealing(
            fun,
            bounds=bounds,
            x0=theta,
            args=func_args,
            maxiter=self.maxiter,
            maxfun=self.maxiter,
            minimizer_kwargs=minimizer_kwargs,
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

    def set_seed(self, seed=None, **kwargs):
        # Set the seed for the global optimizer
        super().set_seed(seed=seed, **kwargs)
        # Set the random seed of the bounds
        if self.bounds is not None and hasattr(self.bounds, "set_seed"):
            self.bounds.set_seed(seed=seed, **kwargs)
        return self

    def set_dtype(self, dtype, **kwargs):
        # Set the dtype for the global optimizer
        super().set_dtype(dtype=dtype, **kwargs)
        # Set the data type of the bounds
        if self.bounds is not None and hasattr(self.bounds, "set_dtype"):
            self.bounds.set_dtype(dtype=dtype, **kwargs)
        return self

    def update_arguments(
        self,
        bounds=None,
        maxiter=None,
        parallel=None,
        jac=None,
        opt_kwargs=None,
        local_kwargs=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's dual_annealing.
            local_kwargs: dict
                A dictionary with the arguments and keywords given
                to SciPy's local minimizer.
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
        if bounds is not None:
            self.bounds = bounds.copy()
        # Set the arguments for the parent class
        super().update_arguments(
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
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
            bounds=self.bounds,
            maxiter=self.maxiter,
            jac=self.jac,
            parallel=self.parallel,
            opt_kwargs=self.opt_kwargs,
            local_kwargs=self.local_kwargs,
            seed=self.seed,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AnneallingTransOptimizer(AnneallingOptimizer):
    """
    The simulated annealing optimizer used for optimzing
    the objective functionwrt. the hyperparameters.
    The simulated annealing optimizer is a wrapper to
    SciPy's dual_annealing.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
    No local optimizer is given to this optimizer.
    The local optimizer is set by keywords in the local_kwargs and
    it uses SciPy's minimizer.
    This simulated annealing optimizer uses variable transformation of
    the hyperparameters to search the space.
    """

    def __init__(
        self,
        bounds=VariableTransformation(),
        maxiter=5000,
        jac=True,
        parallel=False,
        opt_kwargs={},
        local_kwargs={},
        seed=None,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the global optimizer.

        Parameters:
            bounds: VariableTransformation class
                A class of the variable transformation of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given to
                SciPy's dual_annealing.
            local_kwargs: dict
                A dictionary with the arguments and keywords given to
                SciPy's local minimizer.
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set default arguments for SciPy's dual_annealing
        self.opt_kwargs = dict(
            initial_temp=5230.0,
            restart_temp_ratio=2e-05,
            visit=2.62,
            accept=-5.0,
            no_local_search=False,
        )
        # Set default arguments for SciPy's local minimizer
        self.local_kwargs = dict(options={})
        # Set all the arguments
        self.update_arguments(
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=parallel,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    def run(self, func, theta, parameters, model, X, Y, pdis, **kwargs):
        # Get the function arguments for the wrappers
        func_args_w = self.get_wrapper_arguments(
            func,
            parameters,
            model,
            X,
            Y,
            pdis,
            jac=self.jac,
            **kwargs,
        )
        # Set the minimizer kwargs
        minimizer_kwargs = dict(jac=False, **self.local_kwargs)
        # Make boundary conditions
        bounds = self.make_bounds(parameters, use_array=True, transformed=True)
        # Do the dual simulated annealing
        sol = dual_annealing(
            self.func_vartrans,
            bounds=bounds,
            x0=theta,
            args=func_args_w,
            maxiter=self.maxiter,
            maxfun=self.maxiter,
            minimizer_kwargs=minimizer_kwargs,
            **self.opt_kwargs,
        )
        sol = self.get_final_solution(sol, func, parameters, model, X, Y, pdis)
        # Retransform hyperparameters
        sol = self.transform_solution(sol)
        return sol

    def update_arguments(
        self,
        bounds=None,
        maxiter=None,
        jac=None,
        parallel=None,
        opt_kwargs=None,
        local_kwargs=None,
        seed=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the optimizer with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            bounds: VariableTransformation class
                A class of the variable transformation of the hyperparameters.
            maxiter: int
                The maximum number of evaluations or iterations
                the global optimizer can use.
            jac: bool
                Whether to use the gradient of the objective function
                wrt. the hyperparameters.
            parallel: bool
                Whether to calculate the grid points in parallel
                over multiple CPUs.
                This is not implemented for this method.
            opt_kwargs: dict
                A dictionary with the arguments and keywords given to
                SciPy's dual_annealing.
            local_kwargs: dict
                A dictionary with the arguments and keywords given to
                SciPy's local minimizer.
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
        super().update_arguments(
            bounds=bounds,
            maxiter=maxiter,
            jac=jac,
            parallel=False,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
            seed=seed,
            dtype=dtype,
        )
        if bounds is not None:
            if not isinstance(bounds, VariableTransformation):
                raise ValueError(
                    "A variable transformation as bounds has to be used!"
                )
        return self

    def func_vartrans(self, ti, fun, parameters, func_args=(), **kwargs):
        """
        Objective function called for simulated annealing,
        where hyperparameters are transformed.
        """
        theta = self.reverse_trasformation(ti, parameters)
        return fun(theta, *func_args)

    def reverse_trasformation(self, ti, parameters, **kwargs):
        """
        Transform the variable transformed hyperparameters back
        to hyperparameter log-space.
        """
        ti = where(
            ti < 1.0,
            where(ti > 0.0, ti, self.bounds.eps),
            1.00 - self.bounds.eps,
        )
        t = self.make_hp(ti, parameters)
        theta = self.bounds.reverse_trasformation(t, use_array=True)
        return theta

    def transform_solution(self, sol, **kwargs):
        """
        Retransform the variable transformed hyperparameters in
        the solution back to hyperparameter log-space.
        """
        sol["x"] = self.bounds.reverse_trasformation(sol["hp"], use_array=True)
        sol["hp"] = self.bounds.reverse_trasformation(
            sol["hp"],
            use_array=False,
        )
        return sol

    def get_wrapper_arguments(
        self,
        func,
        parameters,
        model,
        X,
        Y,
        pdis,
        jac,
        **kwargs,
    ):
        "Get the function arguments for the wrappers."
        # Get the function arguments
        func_args = self.get_func_arguments(
            parameters,
            model,
            X,
            Y,
            pdis,
            jac=False,
            **kwargs,
        )
        # Get the function that evaluate the objective function
        fun = self.get_fun(func)
        # Get the function arguments for the wrappers
        func_args_w = (fun, parameters, func_args)
        return func_args_w
