from numpy import where
from ase.parallel import world
from .improvedneb import ImprovedTangentNEB


class MaxEWNEB(ImprovedTangentNEB):
    """
    The maximum energy-weighted Nudged Elastic Band method implementation.
    The energy-weighted method uses energy weighting to calculate the
    spring constants.
    The maximum energy subtracted by the energy difference (dE) is used as
    the reference energy for the spring constants.
    """

    def __init__(
        self,
        images,
        k=0.1,
        kl_scale=0.1,
        dE=0.01,
        climb=False,
        remove_rotation_and_translation=False,
        mic=True,
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
            dE: float
                The energy difference between the maximum energy
                and the used reference energy.
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
            save_properties=save_properties,
            parallel=parallel,
            comm=comm,
            **kwargs
        )
        self.kl_scale = kl_scale
        self.dE = dE

    def get_spring_constants(self, **kwargs):
        # Get the spring constants
        energies = self.get_energies()
        # Get the maximum energy
        emax = energies.max()
        # Calculate the reference energy
        e0 = emax - self.dE
        # Calculate the weighted spring constants
        k_l = self.k * self.kl_scale
        if e0 < emax:
            a = (emax - energies[:-1]) / (emax - e0)
            k = where(a < 1.0, (1.0 - a) * self.k + a * k_l, k_l)
        else:
            k = k_l
        return k
