from .local import LocalOptimizer
from ase.parallel import world, broadcast
from ase.optimize import FIRE
from numpy import asarray
from ..structures.neb import OriginalNEB


class LocalNEB(LocalOptimizer):
    """
    The LocalNEB is used to run a local optimization of NEB.
    The LocalNEB is applicable to be used with active learning.
    """

    def __init__(
        self,
        optimizable,
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        """
        Initialize the OptimizerMethod instance.

        Parameters:
            optimizable: NEB instance
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
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the parameters
        self.update_arguments(
            optimizable=optimizable,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def update_optimizable(self, structures, **kwargs):
        # Get the positions of the NEB images
        positions = [image.get_positions() for image in structures[1:-1]]
        positions = asarray(positions).reshape(-1, 3)
        # Set the positions of the NEB images
        self.optimizable.set_positions(positions)
        # Find the minimum path length if possible and requested
        if isinstance(self.optimizable, OriginalNEB):
            self.optimizable.permute_images()
        # Reset the optimization
        self.reset_optimization()
        return self

    def get_structures(
        self,
        get_all=True,
        properties=[],
        allow_calculation=True,
        **kwargs,
    ):
        # Get only the first image
        if not get_all:
            return self.copy_atoms(
                self.optimizable.images[0],
                allow_calculation=False,
                **kwargs,
            )
        # Get all the images
        if self.is_parallel_used():
            return self.get_structures_parallel(
                properties=properties,
                allow_calculation=allow_calculation,
                **kwargs,
            )
        structures = [
            self.copy_atoms(
                self.optimizable.images[0], allow_calculation=False, **kwargs
            )
        ]
        structures += [
            self.copy_atoms(
                image,
                properties=properties,
                allow_calculation=allow_calculation,
                **kwargs,
            )
            for image in self.optimizable.images[1:-1]
        ]
        structures += [
            self.copy_atoms(
                self.optimizable.images[-1], allow_calculation=False, **kwargs
            )
        ]
        return structures

    def get_structures_parallel(
        self,
        properties=[],
        allow_calculation=True,
        **kwargs,
    ):
        "Get the structures in parallel."
        # Get the initial structure
        structures = [
            self.copy_atoms(
                self.optimizable.images[0],
                allow_calculation=False,
                **kwargs,
            )
        ]
        # Get the moving images in parallel
        for i, image in enumerate(self.optimizable.images[1:-1]):
            root = i % self.size
            if self.rank == root:
                image = self.copy_atoms(
                    image,
                    properties=properties,
                    allow_calculation=allow_calculation,
                    **kwargs,
                )
            structures.append(broadcast(image, root=root, comm=self.comm))
        # Get the final structure
        structures.append(
            self.copy_atoms(
                self.optimizable.images[-1],
                allow_calculation=False,
                **kwargs,
            )
        )
        return structures

    def get_candidates(self, **kwargs):
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
        optimizable=None,
        local_opt=None,
        local_opt_kwargs={},
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
            optimizable: NEB instance
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
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            optimizable=optimizable,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            optimizable=self.optimizable,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
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
