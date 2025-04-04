from .method import OptimizerMethod
from ase.parallel import world, broadcast
from numpy import argmin, inf, max as max_


class ParallelOptimizer(OptimizerMethod):
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
        The ParallelOptimizer is used to run an optimization in parallel.
        The ParallelOptimizer is applicable to be used with
        active learning.

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
            for method, structure in zip(self.methods, structures):
                method.update_optimizable(structure, **kwargs)
        else:
            self.method.update_optimizable(structures, **kwargs)
            self.methods = [self.method.copy() for _ in range(self.chains)]
        self.reset_optimization()
        return self

    def get_optimizable(self, **kwargs):
        return self.method.get_optimizable(**kwargs)

    def get_structures(self, get_all=True, **kwargs):
        if get_all:
            return [
                method.get_structures(get_all=get_all)
                for method in self.methods
            ]
        return self.method.get_structures(get_all=get_all, **kwargs)

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
        # Make list of properties
        structures = [None] * self.chains
        candidates = [[]] * self.chains
        converged = [False] * self.chains
        used_steps = [self.steps] * self.chains
        values = [inf] * self.chains
        # Run the optimizations
        for chain, method in enumerate(self.methods):
            root = chain % self.size
            if self.rank == root:
                # Set the random seed
                self.change_seed(chain)
                method.set_seed(self.seed)
                # Run the optimization
                converged[chain] = method.run(
                    fmax=fmax,
                    steps=steps,
                    max_unc=max_unc,
                    dtrust=dtrust,
                    **kwargs,
                )
                # Update the number of steps
                used_steps[chain] += method.get_number_of_steps()
                # Get the structures
                structures[chain] = method.get_structures()
                # Get the candidates
                candidates[chain] = method.get_candidates()
                # Get the values
                if self.method.is_energy_minimized():
                    values[chain] = method.get_potential_energy()
                else:
                    values[chain] = method.get_fmax()
        # Broadcast the saved instances
        for chain in range(self.chains):
            root = chain % self.size
            structures[chain] = broadcast(
                structures[chain],
                root=root,
                comm=self.comm,
            )
            candidates_tmp = broadcast(
                [
                    self.copy_atoms(candidate)
                    for candidate in candidates[chain]
                ],
                root=root,
                comm=self.comm,
            )
            if self.rank != root:
                candidates[chain] = candidates_tmp
            converged[chain] = broadcast(
                converged[chain],
                root=root,
                comm=self.comm,
            )
            used_steps[chain] = broadcast(
                used_steps[chain],
                root=root,
                comm=self.comm,
            )
            values[chain] = broadcast(
                values[chain],
                root=root,
                comm=self.comm,
            )
        # Set the candidates
        self.candidates = []
        for candidate_inner in candidates:
            for candidate in candidate_inner:
                self.candidates.append(candidate)
        # Check the minimum value
        i_min = argmin(values)
        self.method = self.method.update_optimizable(structures[i_min])
        self.steps = max_(used_steps)
        # Check if the optimization is converged
        self._converged = self.check_convergence(
            converged=converged[i_min],
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
        if parallel_run is not None:
            self.parallel_run = parallel_run
            self.check_parallel()
        if chains is not None:
            self.chains = chains
        elif not hasattr(self, "chains"):
            if self.parallel_run:
                chains = comm.size
            else:
                chains = 1
        if method is not None:
            self.method = method.copy()
            self.methods = [method.copy() for _ in range(self.chains)]
            self.setup_optimizable()
        if verbose is not None:
            self.verbose = verbose
        if self.chains % self.size != 0:
            self.message(
                "The number of chains should be divisible by "
                "the number of processors!"
            )
        return self

    def setup_optimizable(self, **kwargs):
        self.optimizable = self.method.get_optimizable()
        self.structures = self.method.get_structures()
        self.candidates = self.method.get_candidates()
        self.reset_optimization()
        return self

    def change_seed(self, chain, **kwargs):
        "Change the random seed for the given chain."
        if isinstance(self.seed, int):
            seed = self.seed + chain
            self.set_seed(seed=seed)
        else:
            [self.rng.random() for _ in range(chain)]
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
