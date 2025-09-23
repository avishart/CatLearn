from .method import OptimizerMethod
from ase.parallel import world, broadcast
from numpy import argmin, inf


class ParallelOptimizer(OptimizerMethod):
    """
    The ParallelOptimizer is used to run an optimization in parallel.
    The ParallelOptimizer is applicable to be used with
    active learning.
    """

    def __init__(
        self,
        method,
        chains=None,
        parallel_run=True,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        """
        Initialize the OptimizerMethod instance.

        Parameters:
            method: OptimizerMethod instance
                The optimization method to be used.
            chains: int (optional)
                The number of optimization that will be run in parallel.
                If not given, the number of chains is set to the number of
                processors if parallel_run is True, otherwise it is set to 1.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
                A different seed is used for each chain if the seed is an
                integer.
        """
        # Set the parameters
        self.update_arguments(
            method=method,
            chains=chains,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def update_optimizable(self, structures, **kwargs):
        if isinstance(structures, list) and len(structures) == self.chains:
            self.method.update_optimizable(structures[0])
            for method, structure in zip(self.methods, structures):
                method.update_optimizable(structure, **kwargs)
        else:
            self.method.update_optimizable(structures, **kwargs)
            for method in self.methods:
                method.update_optimizable(structures, **kwargs)
        # Reset the optimization
        self.setup_optimizable()
        return self

    def get_optimizable(self, **kwargs):
        return self.method.get_optimizable(**kwargs)

    def get_structures(
        self,
        get_all=True,
        properties=[],
        allow_calculation=True,
        **kwargs,
    ):
        if not get_all:
            return self.method.get_structures(
                get_all=get_all,
                properties=properties,
                allow_calculation=allow_calculation,
                **kwargs,
            )
        structures = []
        for chain, method in enumerate(self.methods):
            root = chain % self.size
            if self.rank == root:
                # Get the structure
                structure = method.get_structures(
                    properties=properties,
                    allow_calculation=allow_calculation,
                    **kwargs,
                )
            else:
                structure = None
            # Broadcast the structure
            structures.append(
                broadcast(
                    structure,
                    root=root,
                    comm=self.comm,
                )
            )
        return structures

    def get_candidates(self, **kwargs):
        candidates = []
        for chain, method in enumerate(self.methods):
            root = chain % self.size
            if self.rank == root:
                # Get the candidate(s)
                candidates_tmp = [
                    candidate for candidate in method.get_candidates(**kwargs)
                ]
            else:
                candidates_tmp = []
            # Broadcast the candidates
            for candidate in broadcast(
                candidates_tmp,
                root=root,
                comm=self.comm,
            ):
                candidates.append(candidate)
        return candidates

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
        # Run the optimizations
        converged_list = [
            (
                method.run(
                    fmax=fmax,
                    steps=steps,
                    max_unc=max_unc,
                    dtrust=dtrust,
                    **kwargs,
                )
                if self.rank == chain % self.size
                else False
            )
            for chain, method in enumerate(self.methods)
        ]
        # Save the structures, values, and used steps
        structures = []
        values = []
        for chain, method in enumerate(self.methods):
            root = chain % self.size
            if self.rank == root:
                # Get the structure
                structure = method.get_structures()
                # Get the value
                if self.method.is_energy_minimized():
                    value = method.get_potential_energy()
                else:
                    value = method.get_fmax()
            else:
                structure = None
                value = inf
            # Broadcast the structure
            structures.append(
                broadcast(
                    structure,
                    root=root,
                    comm=self.comm,
                )
            )
            # Broadcast the values
            values.append(
                broadcast(
                    value,
                    root=root,
                    comm=self.comm,
                )
            )
        # Get the number of steps
        self.steps += max(
            [
                broadcast(
                    method.get_number_of_steps(),
                    root=chain % self.size,
                    comm=self.comm,
                )
                for chain, method in enumerate(self.methods)
            ]
        )
        # Find the best optimization
        chain_min = argmin(values)
        root = chain_min % self.size
        # Broadcast whether the optimization is converged
        converged = broadcast(
            converged_list[chain_min],
            root=root,
            comm=self.comm,
        )
        # Get the best structure and update the method
        structure = structures[chain_min]
        self.method = self.method.update_optimizable(structure)
        self.optimizable = self.method.get_optimizable()
        # Check if the optimization is converged
        self._converged = self.check_convergence(
            converged=converged,
            max_unc=max_unc,
            dtrust=dtrust,
            unc_convergence=unc_convergence,
        )
        return self._converged

    def set_calculator(self, calculator, copy_calc=False, **kwargs):
        self.method.set_calculator(calculator, copy_calc=copy_calc, **kwargs)
        for method in self.methods:
            method.set_calculator(calculator, copy_calc=copy_calc, **kwargs)
        return self

    def is_energy_minimized(self):
        return self.method.is_energy_minimized()

    def is_parallel_allowed(self):
        return True

    def update_arguments(
        self,
        method=None,
        chains=None,
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
            method: OptimizerMethod instance
                The optimization method to be used.
            chains: int
                The number of optimization that will be run in parallel.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            optimizable=None,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        # Set the chains
        if chains is not None:
            self.chains = chains
        elif not hasattr(self, "chains"):
            self.chains = self.size
        # Set the method
        if method is not None:
            self.method = method.copy()
            self.methods = [method.copy() for _ in range(self.chains)]
            self.set_seed(seed=self.seed)
            self.setup_optimizable()
        # Check if the method is set correctly
        if len(self.methods) != self.chains:
            self.message(
                "The number of chains should be equal to "
                "the number of methods!",
                is_warning=True,
            )
            self.methods = [method.copy() for _ in range(self.chains)]
            self.set_seed(seed=self.seed)
            self.setup_optimizable()
        # Check if the number of chains is optimal
        if self.chains % self.size != 0:
            self.message(
                "The number of chains should be divisible by "
                "the number of processors!",
                is_warning=True,
            )
        return self

    def set_seed(self, seed=None, **kwargs):
        # Set the seed for the class
        super().set_seed(seed=seed, **kwargs)
        # Set the seed for the method
        if hasattr(self, "method"):
            self.method.set_seed(seed=seed, **kwargs)
            # Set the seed for each method
            if isinstance(seed, int):
                for method in self.methods:
                    method.set_seed(seed=seed, **kwargs)
                    seed += 1
            else:
                for chain, method in enumerate(self.methods):
                    method.set_seed(seed=seed, **kwargs)
                    method.rng.random(size=chain)
        return self

    def setup_optimizable(self, **kwargs):
        self.optimizable = self.method.get_optimizable()
        self.reset_optimization()
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            method=self.method,
            chains=self.chains,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
            seed=self.seed,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
