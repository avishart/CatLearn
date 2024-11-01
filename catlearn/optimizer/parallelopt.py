from .method import OptimizerMethod
from ase.parallel import world, broadcast
import numpy as np


class ParallelOptimizer(OptimizerMethod):
    def __init__(
        self,
        method,
        chains=None,
        parallel_run=True,
        comm=world,
        verbose=False,
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
        """
        # Set the number of chains
        if chains is None:
            if parallel_run:
                chains = comm.size
            else:
                chains = 1
        # Set the parameters
        self.update_arguments(
            method=method,
            chains=chains,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            **kwargs,
        )

    def update_optimizable(self, structures, **kwargs):
        self.method.update_optimizable(structures, **kwargs)
        self.methods = [self.method.copy() for _ in range(self.chains)]
        self.reset_optimization()
        return self

    def get_optimizable(self, **kwargs):
        return self.method.get_optimizable(**kwargs)

    def get_structures(self, get_all=True, **kwargs):
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
        # Set the rank
        rank = 0
        # Make list of properties
        candidates = [None] * self.chains
        converged = [False] * self.chains
        used_steps = [self.steps] * self.chains
        values = [np.inf] * self.chains
        # Run the optimizations
        for chain, method in enumerate(self.methods):
            if self.rank == rank:
                # Set the random seed
                np.random.RandomState(chain + 1)
                # Run the optimization
                method.run(
                    fmax=fmax,
                    steps=steps,
                    max_unc=max_unc,
                    dtrust=dtrust,
                    **kwargs,
                )
                # Update the number of steps
                used_steps[chain] += method.get_number_of_steps()
                # Get the candidates
                candidates[chain] = method.get_candidates()
                # Check if the optimization is converged
                converged[chain] = method.converged()
                # Get the values
                if self.method.is_energy_minimized():
                    values[chain] = method.get_potential_energy()
                else:
                    values[chain] = method.get_fmax()

            # Update the rank
            rank += 1
            if rank == self.size:
                rank = 0
        # Broadcast the saved instances
        rank = 0
        for chain, method in enumerate(self.methods):
            self.methods[chain] = broadcast(
                self.methods[chain],
                root=rank,
                comm=self.comm,
            )
            candidates[chain] = broadcast(
                candidates[chain],
                root=rank,
                comm=self.comm,
            )
            converged[chain] = broadcast(
                converged[chain],
                root=rank,
                comm=self.comm,
            )
            used_steps[chain] = broadcast(
                used_steps[chain],
                root=rank,
                comm=self.comm,
            )
            values[chain] = broadcast(
                values[chain],
                root=rank,
                comm=self.comm,
            )
            # Update the rank
            rank += 1
            if rank == self.size:
                rank = 0
        # Set the candidates
        self.candidates = []
        for candidate_inner in candidates:
            for candidate in candidate_inner:
                self.candidates.append(candidate)
        # Check the minimum value
        i_min = np.argmin(values)
        self.method = self.methods[i_min]
        self.steps = np.max(used_steps)
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
        return self.methods[-1].is_energy_minimized()

    def is_parallel_allowed(self):
        return True

    def update_arguments(
        self,
        method=None,
        chains=None,
        parallel_run=None,
        comm=None,
        verbose=None,
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
        """
        # Set the communicator
        if comm is not None:
            self.comm = comm
            self.rank = comm.rank
            self.size = comm.size
        # Set the verbose
        if verbose is not None:
            self.verbose = verbose
        if chains is not None:
            self.chains = chains
        if method is not None:
            self.method = method.copy()
            self.methods = [method.copy() for _ in range(self.chains)]
            self.setup_optimizable()
        if parallel_run is not None:
            self.parallel_run = parallel_run
            self.check_parallel()
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

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            method=self.method,
            chains=self.chains,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
