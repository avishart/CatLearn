from numpy import arange, asarray, einsum, zeros
from ase.data import covalent_radii
from .baseline import BaselineCalculator
from ..fingerprint.geometry import (
    get_constraints,
    get_full_distance_matrix,
    cosine_cutoff,
)


class RepulsionCalculator(BaselineCalculator):
    """
    A baseline calculator for ASE Atoms instance.
    It uses a repulsive Lennard-Jones potential baseline.
    The power and the scaling of the repulsive Lennard-Jones potential
    can be selected.
    """

    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(
        self,
        reduce_dimensions=True,
        use_forces=True,
        wrap=True,
        include_ncells=True,
        mic=False,
        all_ncells=True,
        cell_cutoff=4.0,
        use_cutoff=True,
        rs_cutoff=3.0,
        re_cutoff=4.0,
        r_scale=0.7,
        power=10,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the baseline calculator.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_forces: bool
                Calculate and store the forces.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            include_ncells: bool
                Include the neighboring cells when calculating the distances.
                The distances will include the neighboring cells.
                include_ncells will replace mic.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
                Either use mic or include_ncells.
            all_ncells: bool
                Use all neighboring cells when calculating the distances.
                cell_cutoff is used to check how many neighboring cells are
                needed.
            cell_cutoff: float
                The cutoff distance for the neighboring cells.
                It is the scaling of the maximum covalent distance.
            use_cutoff: bool
                Whether to use a cutoff function for the inverse distance
                fingerprint.
                The cutoff function is a cosine cutoff function.
            rs_cutoff: float
                The starting distance for the cutoff function being 1.
            re_cutoff: float
                The ending distance for the cutoff function being 0.
                re_cutoff must be larger than rs_cutoff.
            r_scale: float
                The scaling of the covalent radii.
                A smaller value will move the repulsion to a lower distances.
            power: int
                The power of the repulsion.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            reduce_dimensions=reduce_dimensions,
            use_forces=use_forces,
            wrap=wrap,
            include_ncells=include_ncells,
            mic=mic,
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            use_cutoff=use_cutoff,
            rs_cutoff=rs_cutoff,
            re_cutoff=re_cutoff,
            r_scale=r_scale,
            power=power,
            dtype=dtype,
            **kwargs,
        )

    def update_arguments(
        self,
        reduce_dimensions=None,
        use_forces=None,
        wrap=None,
        include_ncells=None,
        mic=None,
        all_ncells=None,
        cell_cutoff=None,
        use_cutoff=None,
        rs_cutoff=None,
        re_cutoff=None,
        r_scale=None,
        power=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_forces: bool
                Calculate and store the forces.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            include_ncells: bool
                Include the neighboring cells when calculating the distances.
                The distances will include the neighboring cells.
                include_ncells will replace mic.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
                Either use mic or include_ncells.
            all_ncells: bool
                Use all neighboring cells when calculating the distances.
                cell_cutoff is used to check how many neighboring cells are
                needed.
            cell_cutoff: float
                The cutoff distance for the neighboring cells.
                It is the scaling of the maximum covalent distance.
            use_cutoff: bool
                Whether to use a cutoff function for the inverse distance
                fingerprint.
                The cutoff function is a cosine cutoff function.
            rs_cutoff: float
                The starting distance for the cutoff function being 1.
            re_cutoff: float
                The ending distance for the cutoff function being 0.
                re_cutoff must be larger than rs_cutoff.
            r_scale: float
                The scaling of the covalent radii.
                A smaller value will move the repulsion to a lower distances.
            power: int
                The power of the repulsion.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        super().update_arguments(
            reduce_dimensions=reduce_dimensions,
            use_forces=use_forces,
            dtype=dtype,
        )
        if wrap is not None:
            self.wrap = wrap
        if include_ncells is not None:
            self.include_ncells = include_ncells
        if mic is not None:
            self.mic = mic
        if all_ncells is not None:
            self.all_ncells = all_ncells
        if cell_cutoff is not None:
            self.cell_cutoff = abs(float(cell_cutoff))
        if use_cutoff is not None:
            self.use_cutoff = use_cutoff
        if rs_cutoff is not None:
            self.rs_cutoff = abs(float(rs_cutoff))
        if re_cutoff is not None:
            self.re_cutoff = abs(float(re_cutoff))
        if r_scale is not None:
            self.r_scale = abs(float(r_scale))
        if power is not None:
            self.power = int(power)
        # Calculate the normalization
        self.set_normalization_constant()
        return self

    def set_normalization_constant(self, **kwargs):
        "Set the normalization constant."
        # Calculate the normalization
        self.c0 = self.r_scale**self.power
        self.c0p = -self.c0 * self.power
        return self

    def get_energy_forces(self, atoms, use_forces=True, **kwargs):
        "Get the energy and forces."
        # Get the not fixed (not masked) atom indices
        not_masked, _ = get_constraints(
            atoms,
            reduce_dimensions=self.reduce_dimensions,
        )
        i_nm = arange(len(not_masked))
        # Check if there are any not masked atoms
        if len(not_masked) == 0:
            if use_forces:
                return 0.0, zeros((len(atoms), 3), dtype=self.dtype)
            return 0.0, None
        # Check what distance method should be used
        (
            use_vector,
            use_include_ncells,
            use_mic,
        ) = self.use_dis_method(pbc=atoms.pbc, use_forces=use_forces, **kwargs)
        # Calculate the inverse distances and their derivatives
        inv_dist, deriv = self.get_inv_dis(
            atoms=atoms,
            not_masked=not_masked,
            i_nm=i_nm,
            use_forces=use_forces,
            use_vector=use_vector,
            use_include_ncells=use_include_ncells,
            use_mic=use_mic,
            **kwargs,
        )
        # Calculate energy
        energy = self.calc_energy(
            inv_dist=inv_dist,
            i_nm=i_nm,
            not_masked=not_masked,
            use_include_ncells=use_include_ncells,
        )
        # Calculate forces
        if use_forces:
            forces = zeros((len(atoms), 3), dtype=self.dtype)
            forces[not_masked] = self.calc_forces(
                inv_dist=inv_dist,
                deriv=deriv,
                i_nm=i_nm,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                **kwargs,
            )
            return energy, forces
        return energy, None

    def calc_energy(
        self,
        inv_dist,
        not_masked,
        i_nm,
        use_include_ncells,
        **kwargs,
    ):
        "Calculate the energy."
        if use_include_ncells:
            inv_dist_p = (inv_dist**self.power).sum(axis=0)
        else:
            inv_dist_p = inv_dist**self.power
        # Take double countings into account
        inv_dist_p[i_nm, not_masked] *= 2.0
        inv_dist_p[:, not_masked] *= 0.5
        energy = self.c0 * inv_dist_p.sum()
        return energy

    def calc_forces(
        self,
        inv_dist,
        deriv,
        not_masked,
        i_nm,
        use_include_ncells=False,
        **kwargs,
    ):
        "Calculate the forces."
        # Calculate the derivative of the energy
        inv_dist_p = inv_dist ** (self.power - 1)
        # Calculate the forces
        if use_include_ncells:
            forces = einsum("dijc,dij->ic", deriv, inv_dist_p)
        else:
            forces = einsum("ijc,ij->ic", deriv, inv_dist_p)
        forces *= self.c0p
        return forces

    def get_inv_dis(
        self,
        atoms,
        not_masked,
        i_nm,
        use_forces,
        use_vector,
        use_include_ncells,
        use_mic,
        **kwargs,
    ):
        """
        Get the inverse distances and their derivatives.

        Parameters:
            atoms: ase.Atoms
                The atoms object.
            not_masked: list
                The indices of the atoms that are not masked.
            i_nm: list
                The indices of the atoms that are not masked.
            use_forces: bool
                Whether to calculate the forces.
            use_vector: bool
                Whether to use the vector of the distances.
            use_include_ncells: bool
                Whether to include the neighboring cells when calculating
                the distances.
            use_mic: bool
                Whether to use the minimum image convention.

        Returns:
            inv_dist: array
                The inverse distances.
            deriv: array
                The derivatives of the inverse distances.
        """
        # Calculate the distances
        dist, dist_vec = self.get_distances(
            atoms=atoms,
            not_masked=not_masked,
            use_vector=use_vector,
            use_include_ncells=use_include_ncells,
            use_mic=use_mic,
            **kwargs,
        )
        # Get the covalent radii
        cov_dis = self.get_covalent_distances(
            atoms.get_atomic_numbers(),
            not_masked,
        )
        # Add a small number to avoid division by zero
        dist += self.eps
        # Check if the distances should be included in the neighboring cells
        if use_include_ncells:
            # Calculate the inverse distances
            inv_dist = cov_dis[None, ...] / dist
            # Remove self interaction
            inv_dist[0, i_nm, not_masked] = 0.0
        else:
            # Calculate the inverse distances
            inv_dist = cov_dis / dist
            # Remove self interaction
            inv_dist[i_nm, not_masked] = 0.0
        # Calculate the derivatives
        if use_forces:
            deriv = dist_vec * (inv_dist / (dist**2))[..., None]
        else:
            deriv = None
        # Calculate the cutoff function
        if self.use_cutoff:
            inv_dist, deriv = cosine_cutoff(
                inv_dist,
                deriv,
                rs_cutoff=self.rs_cutoff,
                re_cutoff=self.re_cutoff,
                eps=self.eps,
            )
        return inv_dist, deriv

    def get_distances(
        self,
        atoms,
        not_masked,
        use_vector,
        use_include_ncells,
        use_mic,
        **kwargs,
    ):
        "Calculate the distances."
        dist, dist_vec = get_full_distance_matrix(
            atoms=atoms,
            not_masked=not_masked,
            use_vector=use_vector,
            wrap=self.wrap,
            include_ncells=use_include_ncells,
            all_ncells=self.all_ncells,
            mic=use_mic,
            cell_cutoff=self.cell_cutoff,
            dtype=self.dtype,
            **kwargs,
        )
        return dist, dist_vec

    def get_covalent_distances(self, atomic_numbers, not_masked):
        "Get the covalent distances of the atoms."
        cov_dis = covalent_radii[atomic_numbers]
        return asarray(cov_dis + cov_dis[not_masked, None], dtype=self.dtype)

    def use_dis_method(self, pbc, use_forces, **kwargs):
        """
        Check what distance method should be used.

        Parameters:
            pbc: bool
                The periodic boundary conditions.
            use_forces: bool
                Whether to calculate the forces.

        Returns:
            use_vector: bool
                Whether to use the vector of the distances.
            use_include_ncells: bool
                Whether to include the neighboring cells when calculating
                the distances.
            use_mic: bool
                Whether to use the minimum image convention.
        """
        if not pbc.any():
            return use_forces, False, False
        if self.include_ncells:
            return True, True, False
        if self.mic:
            return True, False, True
        return use_forces, False, False

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            reduce_dimensions=self.reduce_dimensions,
            use_forces=self.use_forces,
            wrap=self.wrap,
            include_ncells=self.include_ncells,
            mic=self.mic,
            all_ncells=self.all_ncells,
            cell_cutoff=self.cell_cutoff,
            use_cutoff=self.use_cutoff,
            rs_cutoff=self.rs_cutoff,
            re_cutoff=self.re_cutoff,
            r_scale=self.r_scale,
            power=self.power,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
