from ase.parallel import world
from ase.io import read
from ase.optimize import FIRE
from .localneb import LocalNEB
from .sequential import SequentialOptimizer
from ..structures.neb import ImprovedTangentNEB, make_interpolation


class LocalCINEB(SequentialOptimizer):
    def __init__(
        self,
        start,
        end,
        neb_method=ImprovedTangentNEB,
        neb_kwargs={},
        n_images=15,
        climb=True,
        neb_interpolation="linear",
        neb_interpolation_kwargs={},
        reuse_ci_path=False,
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
            start : Atoms instance or ASE Trajectory file.
                The Atoms must have the calculator attached with energy.
                Initial end-point of the NEB path.
            end : Atoms instance or ASE Trajectory file.
                The Atoms must have the calculator attached with energy.
                Final end-point of the NEB path.
            neb_method : NEB class object or str
                The NEB implemented class object used for the ML-NEB.
                A string can be used to select:
                - 'improvedtangentneb' (default)
                - 'ewneb'
            neb_kwargs : dict
                A dictionary with the arguments used in the NEB object
                to create the instance.
                Climb must not be included.
            n_images : int
                Number of images of the path (if not included a path before).
                The number of images include the 2 end-points of the NEB path.
            climb : bool
                Whether to use the climbing image in the NEB.
                It is strongly recommended to have climb=True.
            neb_interpolation : str
                The interpolation method used to create the NEB path.
                The default is 'linear'.
            neb_interpolation_kwargs : dict
                The keyword arguments for the interpolation method.
            reuse_ci_path : bool
                Whether to remove the non-climbing image method when the NEB
                without climbing image is converged.
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
        # Save the end points for creating the NEB
        self.setup_endpoints(start, end)
        # Build the optimizer methods and NEB within
        methods = self.build_method(
            neb_method,
            neb_kwargs=neb_kwargs,
            climb=climb,
            n_images=n_images,
            neb_interpolation=neb_interpolation,
            neb_interpolation_kwargs=neb_interpolation_kwargs,
            reuse_ci_path=reuse_ci_path,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            **kwargs,
        )
        # Set the parameters
        self.update_arguments(
            methods=methods,
            remove_methods=reuse_ci_path,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            **kwargs,
        )

    def setup_endpoints(self, start, end, **kwargs):
        """
        Setup the start and end points for the NEB calculation.
        """
        # Load the start and end points from trajectory files
        if isinstance(start, str):
            start = read(start)
        if isinstance(end, str):
            end = read(end)
        # Save the start point with calculators
        start.get_forces()
        self.start = self.copy_atoms(start)
        # Save the end point with calculators
        end.get_forces()
        self.end = self.copy_atoms(end)
        return self

    def setup_neb(
        self,
        neb_method,
        neb_kwargs={},
        climb=True,
        n_images=15,
        k=3.0,
        remove_rotation_and_translation=False,
        mic=True,
        neb_interpolation="linear",
        neb_interpolation_kwargs={},
        parallel=False,
        comm=None,
        **kwargs,
    ):
        """
        Setup the NEB instance.
        """
        # Create the neb method if it is a string
        if neb_method is None:
            neb_method = ImprovedTangentNEB
        elif isinstance(neb_method, str):
            if neb_method.lower() == "improvedtangentneb":
                neb_method = ImprovedTangentNEB
            elif neb_method.lower() == "ewneb":
                from ..structures.neb.ewneb import EWNEB

                neb_method = EWNEB
            else:
                raise Exception(
                    "The NEB method {} is not implemented.".format(neb_method)
                )
        self.neb_method = neb_method
        # Create default dictionary for creating the NEB
        self.neb_kwargs = dict(
            k=k,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            save_properties=False,
            parallel=parallel,
            world=comm,
        )
        # Save the dictionary for creating the NEB
        self.neb_kwargs.update(neb_kwargs)
        # Save the number of images
        self.n_images = n_images
        # Save the instances for creating the NEB interpolation
        self.neb_interpolation = neb_interpolation
        # Create default dictionary for creating the NEB interpolation
        self.neb_interpolation_kwargs = dict(
            mic=mic,
            remove_rotation_and_translation=remove_rotation_and_translation,
        )
        # Save the dictionary for creating the NEB interpolation
        self.neb_interpolation_kwargs.update(neb_interpolation_kwargs)
        # Make the images for the NEB from the interpolation
        images = make_interpolation(
            start=self.start,
            end=self.end,
            n_images=self.n_images,
            method=self.neb_interpolation,
            **self.neb_interpolation_kwargs,
        )
        # Create the NEB
        neb = self.neb_method(images, climb=climb, **self.neb_kwargs)
        return neb

    def build_method(
        self,
        neb_method,
        neb_kwargs={},
        climb=True,
        n_images=15,
        k=3.0,
        remove_rotation_and_translation=False,
        mic=True,
        neb_interpolation="linear",
        neb_interpolation_kwargs={},
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        **kwargs,
    ):
        "Build the optimization method."
        # Save the instances for creating the local optimizer
        self.local_opt = local_opt
        self.local_opt_kwargs = local_opt_kwargs
        # Save the instances for creating the NEB
        self.climb = climb
        # Setup NEB without climbing image
        neb_noclimb = self.setup_neb(
            neb_method=neb_method,
            neb_kwargs=neb_kwargs,
            climb=False,
            n_images=n_images,
            k=k,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            neb_interpolation=neb_interpolation,
            neb_interpolation_kwargs=neb_interpolation_kwargs,
            parallel=parallel_run,
            comm=comm,
            **kwargs,
        )
        # Build the optimizer method without climbing image
        method_noclimb = LocalNEB(
            neb_noclimb,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
        )
        # Return the method without climbing image
        methods = [method_noclimb]
        if not climb:
            return methods
        # Setup NEB with climbing image
        neb_climb = self.setup_neb(
            neb_method=neb_method,
            neb_kwargs=neb_kwargs,
            climb=True,
            n_images=n_images,
            k=k,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            neb_interpolation=neb_interpolation,
            neb_interpolation_kwargs=neb_interpolation_kwargs,
            parallel=parallel_run,
            comm=comm,
            **kwargs,
        )
        # Build the optimizer method with climbing image
        method_climb = LocalNEB(
            neb_climb,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
        )
        # Return the without and with climbing image
        methods.append(method_climb)
        return methods

    def is_energy_minimized(self):
        return self.methods[-1].is_energy_minimized()

    def is_parallel_allowed(self):
        return True

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            start=self.start,
            end=self.end,
            neb_method=self.neb_method,
            neb_kwargs=self.neb_kwargs,
            n_images=self.n_images,
            climb=self.climb,
            neb_interpolation=self.neb_interpolation,
            neb_interpolation_kwargs=self.neb_interpolation_kwargs,
            reuse_ci_path=self.remove_methods,
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
