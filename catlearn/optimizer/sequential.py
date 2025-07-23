from .method import OptimizerMethod
from ase.parallel import world


class SequentialOptimizer(OptimizerMethod):
    """
    The SequentialOptimizer is used to run multiple optimizations in
    sequence for a given structure.
    The SequentialOptimizer is applicable to be used with
    active learning.
    """

    def __init__(
        self,
        methods,
        remove_methods=False,
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        """
        Initialize the OptimizerMethod instance.

        Parameters:
            methods: List of OptimizerMethod objects
                The list of optimization methods to be used.
            remove_methods: bool
                Whether to remove the methods that have converged.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the parameters
        self.update_arguments(
            methods=methods,
            remove_methods=remove_methods,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def update_optimizable(self, structures, **kwargs):
        # Update optimizable for the first method
        self.methods[0].update_optimizable(structures, **kwargs)
        # Reset the optimization and update the optimizable
        self.setup_optimizable()
        return self

    def get_optimizable(self):
        return self.method.get_optimizable()

    def get_structures(
        self,
        get_all=True,
        properties=[],
        allow_calculation=True,
        **kwargs,
    ):
        return self.method.get_structures(
            get_all=get_all,
            properties=properties,
            allow_calculation=allow_calculation,
            **kwargs,
        )

    def get_candidates(self, **kwargs):
        return self.method.get_candidates(**kwargs)

    def run(
        self,
        fmax=0.05,
        steps=1000000,
        max_unc=None,
        dtrust=None,
        unc_convergence=None,
        **kwargs,
    ):
        # Check if the optimization can take any steps
        if steps <= 0:
            return self._converged
        # Get number of methods
        n_methods = len(self.methods)
        structures = None
        # Run the optimizations
        for i, self.method in enumerate(self.methods):
            # Update the structures if not the first method
            if i > 0:
                self.method.update_optimizable(structures)
            # Run the optimization
            converged = self.method.run(
                fmax=fmax,
                steps=steps,
                max_unc=max_unc,
                dtrust=dtrust,
                **kwargs,
            )
            # Get the structures
            structures = self.method.get_structures(allow_calculation=False)
            self.optimizable = self.method.get_optimizable()
            # Update the number of steps
            self.steps += self.method.get_number_of_steps()
            steps -= self.method.get_number_of_steps()
            # Check if the optimization is converged
            converged = self.check_convergence(
                converged=converged,
                max_unc=max_unc,
                dtrust=dtrust,
                unc_convergence=unc_convergence,
            )
            if not converged:
                break
            # Check if the complete optimization is converged
            if i + 1 == n_methods:
                self._converged = True
                break
            # Check if any steps are left
            if steps <= 0:
                break
            # Check if the method should be removed
            if self.remove_methods and i + 1 < n_methods:
                self.methods = self.methods[1:]
        return self._converged

    def set_calculator(self, calculator, copy_calc=False, **kwargs):
        self.method.set_calculator(calculator, copy_calc=copy_calc, **kwargs)
        for method in self.methods:
            method.set_calculator(calculator, copy_calc=copy_calc, **kwargs)
        return self

    def setup_optimizable(self, **kwargs):
        self.method = self.methods[0]
        self.optimizable = self.method.get_optimizable()
        self.reset_optimization()
        return self

    def is_energy_minimized(self):
        return self.methods[-1].is_energy_minimized()

    def is_parallel_allowed(self):
        return False

    def update_arguments(
        self,
        methods=None,
        remove_methods=None,
        parallel_run=None,
        comm=None,
        verbose=None,
        seed=None,
        **kwargs,
    ):
        """
        Update the instance with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            methods: List of OptimizerMethod objects
                The list of optimization methods to be used.
            remove_methods: bool
                Whether to remove the methods that have converged.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the methods
        if methods is not None:
            self.methods = methods
            self.setup_optimizable()
        # Set the remove methods
        if remove_methods is not None:
            self.remove_methods = remove_methods
        # Set the parameters in the parent class
        super().update_arguments(
            optimizable=None,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        return self

    def set_seed(self, seed=None, **kwargs):
        # Set the seed for the class
        super().set_seed(seed=seed, **kwargs)
        # Set the seed for each method
        for method in self.methods:
            method.set_seed(seed=seed, **kwargs)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            methods=self.methods,
            remove_methods=self.remove_methods,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
