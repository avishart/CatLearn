from .method import OptimizerMethod
from ase.parallel import world


class SequentialOptimizer(OptimizerMethod):
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
        The SequentialOptimizer is used to run multiple optimizations in
        sequence for a given structure.
        The SequentialOptimizer is applicable to be used with
        active learning.

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
        self.optimizable = self.methods[0].get_optimizable()
        # Reset the optimization and update the optimizable
        self.setup_optimizable()
        return self

    def get_optimizable(self):
        return self.optimizable

    def get_structures(self, get_all=True, **kwargs):
        if isinstance(self.structures, list):
            if not get_all:
                return self.copy_atoms(self.structures[0])
            return [self.copy_atoms(struc) for struc in self.structures]
        return self.copy_atoms(self.structures)

    def get_candidates(self, **kwargs):
        return self.candidates

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
        # Run the optimizations
        for i, method in enumerate(self.methods):
            # Update the structures if not the first method
            if i > 0:
                method.update_optimizable(self.structures)
            # Run the optimization
            converged = method.run(
                fmax=fmax,
                steps=steps,
                max_unc=max_unc,
                dtrust=dtrust,
                **kwargs,
            )
            # Get the structures
            self.optimizable = method.get_optimizable()
            self.structures = method.get_structures()
            self.candidates = method.get_candidates()
            # Update the number of steps
            self.steps += method.get_number_of_steps()
            steps -= method.get_number_of_steps()
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
        for method in self.methods:
            method.set_calculator(calculator, copy_calc=copy_calc, **kwargs)
        return self

    def setup_optimizable(self, **kwargs):
        self.optimizable = self.methods[0].get_optimizable()
        self.structures = self.methods[0].get_structures()
        self.candidates = self.methods[0].get_candidates()
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
        # Set the communicator
        if comm is not None:
            self.comm = comm
            self.rank = comm.rank
            self.size = comm.size
        elif not hasattr(self, "comm"):
            self.comm = None
            self.rank = 0
            self.size = 1
        # Set the seed
        if seed is not None or not hasattr(self, "seed"):
            self.set_seed(seed)
        # Set the verbose
        if verbose is not None:
            self.verbose = verbose
        if remove_methods is not None:
            self.remove_methods = remove_methods
        if methods is not None:
            self.methods = methods
            self.setup_optimizable()
        if parallel_run is not None:
            self.parallel_run = parallel_run
            self.check_parallel()
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
