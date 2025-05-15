from numpy import where
from .repulsive import RepulsionCalculator


class BornRepulsionCalculator(RepulsionCalculator):
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
        cell_cutoff=2.0,
        r_scale=0.8,
        power=2,
        rs1_cross=0.9,
        k_scale=1.0,
        dtype=float,
        **kwargs,
    ):
        """
        A baseline calculator for ASE atoms object.
        It uses a repulsive Lennard-Jones potential baseline.
        The power and the scaling of the repulsive Lennard-Jones potential
        can be selected.

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
            r_scale: float
                The scaling of the covalent radii.
                A smaller value will move the repulsion to a lower distances.
                All distances larger than r_scale is cutoff.
            power: int
                The power of the repulsion.
            rs1_cross: float
                The scaled value of the inverse distance with scaling (r_scale)
                that crosses the energy of 1 eV.
            k_scale: float
                The scaling of the repulsion energy after a default scaling
                of the energy is calculated.
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
            r_scale=r_scale,
            power=power,
            rs1_cross=rs1_cross,
            k_scale=k_scale,
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
        r_scale=None,
        power=None,
        rs1_cross=None,
        k_scale=None,
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
            r_scale: float
                The scaling of the covalent radii.
                A smaller value will move the repulsion to a lower distances.
                All distances larger than r_scale is cutoff.
            power: int
                The power of the repulsion.
            rs1_cross: float
                The scaled value of the inverse distance with scaling (r_scale)
                that crosses the energy of 1 eV.
            k_scale: float
                The scaling of the repulsion energy after a default scaling
                of the energy is calculated.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        # Set the arguments
        if rs1_cross is not None:
            self.rs1_cross = abs(float(rs1_cross))
        if k_scale is not None:
            self.k_scale = abs(float(k_scale))
        # Update the arguments of the parent class
        super().update_arguments(
            reduce_dimensions=reduce_dimensions,
            use_forces=use_forces,
            wrap=wrap,
            include_ncells=include_ncells,
            mic=mic,
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            use_cutoff=False,
            rs_cutoff=None,
            re_cutoff=None,
            r_scale=r_scale,
            power=power,
            dtype=dtype,
        )
        return self

    def set_normalization_constant(self, **kwargs):
        # Calculate the normalization
        self.c0 = self.k_scale / ((1.0 / self.rs1_cross - 1.0) ** self.power)
        self.c0p = -self.c0 * self.power * self.r_scale
        return self

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
        # Calculate the inverse distances
        inv_dist, deriv = super().get_inv_dis(
            atoms=atoms,
            not_masked=not_masked,
            i_nm=i_nm,
            use_forces=use_forces,
            use_vector=use_vector,
            use_include_ncells=use_include_ncells,
            use_mic=use_mic,
            **kwargs,
        )
        # Calculate the scaled inverse distances
        inv_dist = self.r_scale * inv_dist - 1.0
        # Use only the repulsive part
        inv_dist = where(inv_dist < 0.0, 0.0, inv_dist)
        return inv_dist, deriv

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
            r_scale=self.r_scale,
            power=self.power,
            dtype=self.dtype,
            rs1_cross=self.rs1_cross,
            k_scale=self.k_scale,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
