from ase.parallel import world
from ase.io import read
from ase.optimize import FIRE
from .localneb import LocalNEB
from .sequential import SequentialOptimizer
from ..structures.neb import (
    AvgEWNEB,
    EWNEB,
    ImprovedTangentNEB,
    OriginalNEB,
    make_interpolation,
)


class LocalCINEB(SequentialOptimizer):
    """
    The LocalCINEB is used to run a local optimization of NEB.
    First, the NEB is run without climbing image.
    Then, the climbing image is started from the converged
    non-climbing images if clim=True.
    The LocalCINEB is applicable to be used with active learning.
    """

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
        start_without_ci=True,
        reuse_ci_path=False,
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
            start: Atoms instance or ASE Trajectory file.
                The Atoms must have the calculator attached with energy.
                Initial end-point of the NEB path.
            end: Atoms instance or ASE Trajectory file.
                The Atoms must have the calculator attached with energy.
                Final end-point of the NEB path.
            neb_method: NEB class object or str
                The NEB implemented class object used for the ML-NEB.
                A string can be used to select:
                - 'improvedtangentneb' (default)
                - 'ewneb'
                - 'avgewneb'
            neb_kwargs: dict
                A dictionary with the arguments used in the NEB object
                to create the instance.
                Climb and images must not be included.
            n_images: int
                Number of images of the path (if not included a path before).
                The number of images include the 2 end-points of the NEB path.
            climb: bool
                Whether to use the climbing image in the NEB.
                It is strongly recommended to have climb=True.
            neb_interpolation: str or list of ASE Atoms or ASE Trajectory file
                The interpolation method used to create the NEB path.
                The string can be:
                - 'linear' (default)
                - 'idpp'
                - 'rep'
                - 'born'
                - 'ends'
                Otherwise, the premade images can be given as a list of
                ASE Atoms.
                A string of the ASE Trajectory file that contains the images
                can also be given.
            neb_interpolation_kwargs: dict
                The keyword arguments for the interpolation method.
                It is only used when the interpolation method is a string.
            start_without_ci: bool
                Whether to start the NEB without the climbing image.
                If True, the NEB path will be optimized without
                the climbing image and afterwards climbing image is used
                if climb=True as well.
                If False, the NEB path will be optimized with the climbing
                image if climb=True as well.
            reuse_ci_path: bool
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
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the verbose
        self.verbose = verbose
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
            start_without_ci=start_without_ci,
            reuse_ci_path=reuse_ci_path,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )
        # Set the parameters
        self.update_arguments(
            methods=methods,
            remove_methods=reuse_ci_path,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
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
        self.start = self.copy_atoms(
            start,
            properties=["forces", "energy"],
            allow_calculation=True,
            **kwargs,
        )
        # Save the end point with calculators
        end.get_forces()
        self.end = self.copy_atoms(
            end,
            properties=["forces", "energy"],
            allow_calculation=True,
            **kwargs,
        )
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
        seed=None,
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
                neb_method = EWNEB
            elif neb_method.lower() == "avgewneb":
                neb_method = AvgEWNEB
            else:
                raise ValueError(
                    "The NEB method {} is not implemented.".format(neb_method)
                )
        self.neb_method = neb_method
        # Create default dictionary for creating the NEB
        self.neb_kwargs = dict(
            k=k,
            remove_rotation_and_translation=remove_rotation_and_translation,
            parallel=parallel,
        )
        if isinstance(neb_method, str) or issubclass(neb_method, OriginalNEB):
            self.neb_kwargs.update(
                dict(
                    use_image_permutation=True,
                    save_properties=True,
                    mic=mic,
                    comm=comm,
                )
            )
        else:
            self.neb_kwargs.update(dict(world=comm))
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
            seed=seed,
        )
        # Save the dictionary for creating the NEB interpolation
        self.neb_interpolation_kwargs.update(neb_interpolation_kwargs)
        # Make the images for the NEB from the interpolation
        images = make_interpolation(
            start=self.start,
            end=self.end,
            n_images=self.n_images,
            method=self.neb_interpolation,
            neb_method=neb_method,
            neb_kwargs=self.neb_kwargs,
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
        start_without_ci=True,
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        "Build the optimization method."
        # Save the instances for creating the local optimizer
        self.local_opt = local_opt
        self.local_opt_kwargs = local_opt_kwargs
        # Save the instances for creating the NEB
        self.climb = climb
        self.start_without_ci = start_without_ci
        # Check if climb and start_without_ci are compatible
        if not start_without_ci and not climb:
            self.message(
                "If start_without_ci is False, climb must be True!"
                "start_without_ci is set to True.",
                is_warning=True,
            )
            self.start_without_ci = True
        # Set the kwargs for setting up the NEB
        setup_neb_kwargs = dict(
            neb_method=neb_method,
            neb_kwargs=neb_kwargs,
            n_images=n_images,
            k=k,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            neb_interpolation=neb_interpolation,
            neb_interpolation_kwargs=neb_interpolation_kwargs,
            parallel=parallel_run,
            comm=comm,
            seed=seed,
            **kwargs,
        )
        # Check if the non-climbing image method should be used
        if self.start_without_ci:
            # Setup NEB without climbing image
            neb_noclimb = self.setup_neb(
                climb=False,
                **setup_neb_kwargs,
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
        else:
            methods = []
        # Setup NEB with climbing image
        neb_climb = self.setup_neb(
            climb=True,
            **setup_neb_kwargs,
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
            start_without_ci=self.start_without_ci,
            reuse_ci_path=self.remove_methods,
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
