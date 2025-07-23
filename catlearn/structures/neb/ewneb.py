from numpy import where
from ase.parallel import world
from .improvedneb import ImprovedTangentNEB


class EWNEB(ImprovedTangentNEB):
    """
    The energy-weighted Nudged Elastic Band method implementation.
    The energy-weighted method uses energy weighting to calculate the spring
    constants.
    See:
        https://doi.org/10.1021/acs.jctc.1c00462
    """

    def __init__(
        self,
        images,
        k=0.1,
        kl_scale=0.1,
        use_minimum=False,
        climb=False,
        remove_rotation_and_translation=False,
        mic=True,
        use_image_permutation=False,
        save_properties=False,
        parallel=False,
        comm=world,
        **kwargs
    ):
        """
        Initialize the NEB instance.

        Parameters:
            images: List of ASE Atoms instances
                The ASE Atoms instances used as the images of the initial path
                that is optimized.
            k: List of floats or float
                The (Nimg-1) spring forces acting between each image.
                In the energy-weighted Nudged Elastic Band method, this spring
                constants are the upper spring constants.
            kl_scale: float
                The scaling factor for the lower spring constants.
            use_minimum: bool
                Whether to use the minimum energy as the reference energy
                for the spring constants.
                If False, the maximum energy is used.
            climb: bool
                Whether to use climbing image in the NEB.
                See:
                    https://doi.org/10.1063/1.1329672
            remove_rotation_and_translation: bool
                Whether to remove rotation and translation in interpolation
                and when predicting forces.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
            use_image_permutation: bool
                Whether to permute images to minimize the path length.
                It assumes a greedy algorithm to find the minimum path length
                by selecting the next image that is closest to the previous
                image.
                It is only used in the initialization of the NEB.
            save_properties: bool
                Whether to save the properties by making a copy of the images.
            parallel: bool
                Whether to run the calculations in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.
        """
        super().__init__(
            images,
            k=k,
            climb=climb,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            use_image_permutation=use_image_permutation,
            save_properties=save_properties,
            parallel=parallel,
            comm=comm,
            **kwargs
        )
        self.kl_scale = kl_scale
        self.use_minimum = use_minimum

    def get_spring_constants(self, **kwargs):
        # Get the energies
        energies = self.get_energies()
        # Get the reference energy
        if self.use_minimum:
            e0 = min([energies[0], energies[-1]])
        else:
            e0 = max([energies[0], energies[-1]])
        # Get the maximum energy
        emax = energies.max()
        # Calculate the weighted spring constants
        k_l = self.k * self.kl_scale
        if e0 < emax:
            a = (emax - energies[:-1]) / (emax - e0)
            k = where(a < 1.0, (1.0 - a) * self.k + a * k_l, k_l)
        else:
            k = k_l
        return k

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            images=self.images,
            k=self.k,
            kl_scale=self.kl_scale,
            use_minimum=self.use_minimum,
            climb=self.climb,
            remove_rotation_and_translation=self.rm_rot_trans,
            mic=self.mic,
            use_image_permutation=self.use_image_permutation,
            save_properties=self.save_properties,
            parallel=self.parallel,
            comm=self.comm,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
