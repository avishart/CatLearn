import numpy as np
from .baseline import BaselineCalculator
from ..fingerprint.geometry import get_full_distance_matrix


class IDPP(BaselineCalculator):

    def __init__(
        self,
        target=[],
        mic=False,
        **kwargs,
    ):
        """
        A baseline calculator for ASE atoms object.
        It uses image dependent pair potential.

        Parameters:
            target: array
                The target distances for the IDPP.
            mic : bool
                Minimum Image Convention (Shortest distances
                when periodic boundary conditions are used).

        See:
            Improved initial guess for minimum energy path calculations.
            Søren Smidstrup, Andreas Pedersen, Kurt Stokbro and Hannes Jónsson
            Chem. Phys. 140, 214106 (2014)
        """
        super().__init__()
        self.update_arguments(
            target=target,
            mic=mic,
            **kwargs,
        )

    def update_arguments(
        self,
        target=None,
        mic=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            target: array
                The target distances for the IDPP.
            mic : bool
                Minimum Image Convention (Shortest distances
                when periodic boundary conditions are used).

        Returns:
            self: The updated object itself.
        """
        if target is not None:
            self.target = target.copy()
        if mic is not None:
            self.mic = mic
        return self

    def get_energy_forces(self, atoms, get_derivatives=True, **kwargs):
        "Get the energy and forces."
        # Get all distances
        dis, dis_vec = get_full_distance_matrix(
            atoms,
            not_masked=None,
            mic=self.mic,
            vector=get_derivatives,
            wrap=False,
        )
        # Get the number of atoms
        n_atoms = len(atoms)
        # Get weights without division by zero
        dis_non = dis.copy()
        dis_non[range(n_atoms), range(n_atoms)] = 1.0
        weights = 1.0 / (dis_non**4)
        # Calculate the energy
        dis_t = dis - self.target
        dis_t2 = dis_t**2
        e = 0.5 * np.sum(weights * dis_t2)
        if get_derivatives:
            # Calculate the forces
            finner = 2.0 * (weights / dis_non) * dis_t2
            finner -= weights * dis_t
            finner = finner / dis_non
            f = np.sum(dis_vec * finner[:, :, None], axis=0)
            return e, f
        return e

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            target=self.target,
            mic=self.mic,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
