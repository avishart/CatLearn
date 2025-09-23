from .baseline import BaselineCalculator
from ..fingerprint.geometry import get_full_distance_matrix


class IDPP(BaselineCalculator):
    """
    A baseline calculator for ASE Atoms instance.
    It uses image dependent pair potential.
    (https://doi.org/10.1063/1.4878664)
    """

    def __init__(
        self,
        target=[],
        wrap=False,
        mic=False,
        use_forces=True,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the baseline calculator.

        Parameters:
            target: array
                The target distances for the IDPP.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            mic: bool
                Minimum Image Convention (Shortest distances
                when periodic boundary conditions are used).
            use_forces: bool
                Calculate and store the forces.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            reduce_dimensions=False,
            target=target,
            wrap=wrap,
            mic=mic,
            use_forces=use_forces,
            dtype=dtype,
        )

    def update_arguments(
        self,
        target=None,
        wrap=None,
        mic=None,
        use_forces=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            target: array
                The target distances for the IDPP.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            mic: bool
                Minimum Image Convention (Shortest distances
                when periodic boundary conditions are used).
            use_forces: bool
                Calculate and store the forces.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        super().update_arguments(
            reduce_dimensions=False,
            use_forces=use_forces,
            dtype=dtype,
        )
        if target is not None:
            self.target = target.copy()
        if wrap is not None:
            self.wrap = wrap
        if mic is not None:
            self.mic = mic
        return self

    def get_energy_forces(self, atoms, use_forces=True, **kwargs):
        "Get the energy and forces."
        # Get all distances
        dis, dis_vec = self.get_distances(
            atoms=atoms,
            use_vector=use_forces,
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
        e = 0.5 * (weights * dis_t2).sum()
        if use_forces:
            # Calculate the forces
            finner = 2.0 * (weights / dis_non) * dis_t2
            finner -= weights * dis_t
            finner = finner / dis_non
            f = (dis_vec * finner[:, :, None]).sum(axis=0)
            return e, f
        return e, None

    def get_distances(
        self,
        atoms,
        use_vector,
        **kwargs,
    ):
        "Calculate the distances."
        dist, dist_vec = get_full_distance_matrix(
            atoms=atoms,
            not_masked=None,
            use_vector=use_vector,
            wrap=self.wrap,
            include_ncells=False,
            all_ncells=False,
            mic=self.mic,
            dtype=self.dtype,
            **kwargs,
        )
        return dist, dist_vec

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            target=self.target,
            wrap=self.wrap,
            mic=self.mic,
            use_forces=self.use_forces,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
