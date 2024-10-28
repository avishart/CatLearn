from .local import LocalOptimizer
from ase.parallel import world
from ase.optimize import FIRE
import numpy as np


class LocalNEB(LocalOptimizer):
    def __init__(
        self,
        neb,
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        **kwargs,
    ):
        """
        The LocalNEB is used to run a local optimization of NEB.
        The LocalNEB is applicable to be used with active learning.

        Parameters:
            neb: NEB instance
                The NEB object to be optimized.
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
        """
        # Set the parameters
        self.update_arguments(
            neb=neb,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            **kwargs,
        )

    def update_optimizable(self, structures, **kwargs):
        # Get the positions of the NEB images
        positions = [image.get_positions() for image in structures[1:-1]]
        positions = np.asarray(positions).reshape(-1, 3)
        # Set the positions of the NEB images
        self.optimizable.set_positions(positions)
        # Reset the optimization
        self.reset_optimization()
        return self

    def get_structures(self):
        return [self.copy_atoms(image) for image in self.optimizable.images]

    def get_candidates(self):
        return self.optimizable.images[1:-1]

    def set_calculator(self, calculator, copy_calc=False, **kwargs):
        if isinstance(calculator, list):
            if len(calculator) != len(self.optimizable.images[1:-1]):
                raise Exception(
                    "The number of calculators should be equal to "
                    "the number of moving images!"
                )
            for image, calc in zip(self.optimizable.images[1:-1], calculator):
                if copy_calc:
                    image.calc = calc.copy()
                else:
                    image.calc = calc
        else:
            for image in self.optimizable.images[1:-1]:
                if copy_calc:
                    image.calc = calculator.copy()
                else:
                    image.calc = calculator
        return self

    def get_calculator(self):
        return [image.calc for image in self.optimizable.images[1:-1]]

    def is_energy_minimized(self):
        return False

    def is_parallel_allowed(self):
        return True

    def update_arguments(
        self,
        neb=None,
        local_opt=None,
        local_opt_kwargs={},
        parallel_run=None,
        comm=None,
        verbose=None,
        **kwargs,
    ):
        """
        Update the instance with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            neb: NEB instance
                The NEB object to be optimized.
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
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
        if neb is not None:
            self.setup_optimizable(neb)
        if local_opt is not None and local_opt_kwargs is not None:
            self.setup_local_optimizer(local_opt, local_opt_kwargs)
        elif local_opt is not None:
            self.setup_local_optimizer(self.local_opt)
        elif local_opt_kwargs is not None:
            self.setup_local_optimizer(self.local_opt, local_opt_kwargs)
        if parallel_run is not None:
            self.parallel_run = parallel_run
            self.check_parallel()
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            neb=self.optimizable,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
