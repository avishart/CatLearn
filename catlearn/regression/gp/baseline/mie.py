from numpy import einsum
from .repulsive import RepulsionCalculator


class MieCalculator(RepulsionCalculator):
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
        use_cutoff=False,
        rs_cutoff=3.0,
        re_cutoff=4.0,
        r_scale=0.7,
        denergy=0.1,
        power_r=8,
        power_a=6,
        dtype=float,
        **kwargs,
    ):
        """
        A baseline calculator for ASE atoms object.
        It uses the Mie potential baseline.
        The power and the scaling of the Mie potential can be selected.

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
            denergy : float
                The dispersion energy of the potential.
            power_r : int
                The power of the repulsive part.
            power_a : int
                The power of the attractive part.
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
            denergy=denergy,
            power_a=power_a,
            power_r=power_r,
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
        denergy=None,
        power_r=None,
        power_a=None,
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
            denergy : float
                The dispersion energy of the potential.
            power_r : int
                The power of the repulsive part.
            power_a : int
                The power of the attractive part.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        # Update the arguments of the class
        if denergy is not None:
            self.denergy = float(denergy)
        if power_r is not None:
            self.power_r = int(power_r)
        if power_a is not None:
            self.power_a = int(power_a)
        # Update the arguments of the parent class
        super().update_arguments(
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
            power=None,
            dtype=dtype,
        )
        return self

    def set_normalization_constant(self, **kwargs):
        # Calculate the normalization
        power_ar = self.power_a / (self.power_r - self.power_a)
        c0 = self.denergy * ((self.power_r / self.power_a) ** power_ar)
        c0 = c0 * (self.power_r / (self.power_r - self.power_a))
        # Calculate the r_scale powers
        self.r_scale_r = c0 * (self.r_scale**self.power_r)
        self.r_scale_a = c0 * (self.r_scale**self.power_a)
        self.power_ar = -self.power_a * self.r_scale_a
        self.power_rr = -self.power_r * self.r_scale_r
        return self

    def calc_energy(
        self,
        inv_dist,
        not_masked,
        i_nm,
        use_include_ncells,
        **kwargs,
    ):
        "Calculate the energy."
        # Get the repulsive part
        if use_include_ncells:
            inv_dist_p = (inv_dist**self.power_r).sum(axis=0)
        else:
            inv_dist_p = inv_dist**self.power_r
        # Take double countings into account
        inv_dist_p[i_nm, not_masked] *= 2.0
        inv_dist_p[:, not_masked] *= 0.5
        energy = self.r_scale_r * inv_dist_p.sum()
        # Get the attractive part
        if use_include_ncells:
            inv_dist_p = (inv_dist**self.power_a).sum(axis=0)
        else:
            inv_dist_p = inv_dist**self.power_a
        # Take double countings into account
        inv_dist_p[i_nm, not_masked] *= 2.0
        inv_dist_p[:, not_masked] *= 0.5
        energy -= self.r_scale_a * inv_dist_p.sum()
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
        # Calculate the derivative of the repulsive energy
        inv_dist_p = inv_dist ** (self.power_r - 1)
        # Calculate the forces
        if use_include_ncells:
            forces = einsum("dijc,dij->ic", deriv, inv_dist_p)
        else:
            forces = einsum("ijc,ij->ic", deriv, inv_dist_p)
        forces *= self.power_rr
        # Calculate the derivative of the attractive energy
        inv_dist_p = inv_dist ** (self.power_a - 1)
        # Calculate the forces
        if use_include_ncells:
            forces -= einsum("dijc,dij->ic", deriv, inv_dist_p)
        else:
            forces -= self.power_ar * einsum("ijc,ij->ic", deriv, inv_dist_p)
        return forces

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
            denergy=self.denergy,
            power_a=self.power_a,
            power_r=self.power_r,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
