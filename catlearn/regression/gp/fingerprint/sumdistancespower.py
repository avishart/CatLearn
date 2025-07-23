from numpy import zeros
from .sumdistances import SumDistances


class SumDistancesPower(SumDistances):
    """
    Fingerprint constructor class that convert an atoms instance into
    a fingerprint instance with vector and derivatives.
    The sum of multiple powers of the inverse distance fingerprint
    constructer class.
    The inverse distances are scaled with covalent radii.
    """

    def __init__(
        self,
        reduce_dimensions=True,
        use_derivatives=True,
        wrap=True,
        include_ncells=False,
        periodic_sum=False,
        periodic_softmax=True,
        mic=False,
        all_ncells=True,
        cell_cutoff=4.0,
        use_cutoff=False,
        rs_cutoff=3.0,
        re_cutoff=4.0,
        dtype=float,
        use_tags=False,
        use_pairs=True,
        reuse_combinations=True,
        power=4,
        use_roots=True,
        **kwargs,
    ):
        """
        Initialize the fingerprint constructor.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Calculate and store derivatives of the fingerprint wrt.
                the cartesian coordinates.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            include_ncells: bool
                Include the neighboring cells when calculating the distances.
                The fingerprint will include the neighboring cells.
                include_ncells will replace periodic_softmax and mic.
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
            periodic_sum: bool
                Use a sum of the distances to neighboring cells
                when periodic boundary conditions are used.
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
            periodic_softmax: bool
                Use a softmax weighting on the distances to neighboring cells
                from the squared distances when periodic boundary conditions
                are used.
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
                mic is faster than periodic_softmax,
                but the derivatives are discontinuous.
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
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            use_tags: bool
                Use the tags of the atoms to identify the atoms as
                another type.
            use_pairs: bool
                Whether to use pairs of elements or use all elements.
            reuse_combinations: bool
                Whether to reuse the combinations of the elements.
                The change in the atomic numbers and tags will be checked
                to see if they are unchanged.
                If False, the combinations are calculated each time.
            power: int
                The power of the inverse distances.
            use_roots: bool
                Whether to use roots of the power elements.
        """
        # Set the arguments
        super().__init__(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            wrap=wrap,
            include_ncells=include_ncells,
            periodic_sum=periodic_sum,
            periodic_softmax=periodic_softmax,
            mic=mic,
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            use_cutoff=use_cutoff,
            rs_cutoff=rs_cutoff,
            re_cutoff=re_cutoff,
            dtype=dtype,
            use_tags=use_tags,
            use_pairs=use_pairs,
            reuse_combinations=reuse_combinations,
            power=power,
            use_roots=use_roots,
            **kwargs,
        )

    def modify_fp(
        self,
        fp,
        g,
        atomic_numbers,
        tags,
        not_masked,
        masked,
        nmi,
        nmj,
        nmi_ind,
        nmj_ind,
        use_include_ncells,
        **kwargs,
    ):
        # Get the indices of the atomic combinations
        split_indices_nm, split_indices = self.element_setup(
            atomic_numbers,
            tags,
            not_masked,
            **kwargs,
        )
        # Get the number of atomic combinations
        if self.use_pairs:
            fp_len = len(split_indices_nm) * len(split_indices)
        else:
            fp_len = len(split_indices_nm)
        # Create the new fingerprint and derivatives
        fp_new = zeros(
            (fp_len, self.power),
            dtype=self.dtype,
        )
        g_new = zeros(
            (
                fp_len,
                self.power,
                3 * len(not_masked),
            ),
            dtype=self.dtype,
        )
        # Loop over the powers
        for p in range(self.power):
            power = p + 1
            if power > 1:
                # Calculate the power of the inverse distances at power > 1
                fp_new[:, p], g_new[:, p] = self.modify_fp_powers(
                    fp=fp,
                    g=g,
                    not_masked=not_masked,
                    masked=masked,
                    nmi=nmi,
                    nmj=nmj,
                    use_include_ncells=use_include_ncells,
                    split_indices_nm=split_indices_nm,
                    split_indices=split_indices,
                    power=power,
                )
            else:
                # Special case for power equal to 1
                fp_new[:, p], g_new[:, p] = self.modify_fp_power1(
                    fp=fp,
                    g=g,
                    not_masked=not_masked,
                    masked=masked,
                    nmi=nmi,
                    nmj=nmj,
                    use_include_ncells=use_include_ncells,
                    split_indices_nm=split_indices_nm,
                    split_indices=split_indices,
                )
        # Reshape fingerprint and derivatives
        fp_new = fp_new.reshape(-1)
        # Return the new fingerprint and derivatives
        if g is not None:
            g_new = g_new.reshape(-1, 3 * len(not_masked))
            return fp_new, g_new
        return fp_new, None

    def modify_fp_power1(
        self,
        fp,
        g,
        not_masked,
        masked,
        nmi,
        nmj,
        use_include_ncells,
        split_indices_nm,
        split_indices,
        **kwargs,
    ):
        """
        Calculate the sum of the inverse distances at power = 1
        for each sets of atomic combinations.
        """
        # Modify the fingerprint
        if self.use_pairs:
            # Use pairs of elements
            fp_new, g_new = self.modify_fp_pairs(
                fp=fp,
                g=g,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                split_indices_nm=split_indices_nm,
                split_indices=split_indices,
                **kwargs,
            )
        else:
            # Use all elements
            fp_new, g_new = self.modify_fp_elements(
                fp=fp,
                g=g,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                split_indices_nm=split_indices_nm,
                **kwargs,
            )
        # Add a small number to avoid division by zero
        fp_new += self.eps
        return fp_new, g_new

    def modify_fp_powers(
        self,
        fp,
        g,
        not_masked,
        masked,
        nmi,
        nmj,
        use_include_ncells,
        split_indices_nm,
        split_indices,
        power,
        **kwargs,
    ):
        """
        Calculate the sum of the inverse distances at power > 1
        for each sets of atomic combinations.
        """
        # Calculate the power of the inverse distances
        fp_new = fp**power
        # Calculate the derivatives
        if g is not None:
            g_new = (fp ** (power - 1))[..., None] * g
        else:
            g_new = None
        # Modify the fingerprint
        if self.use_pairs:
            # Use pairs of elements
            fp_new, g_new = self.modify_fp_pairs(
                fp=fp_new,
                g=g_new,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                split_indices_nm=split_indices_nm,
                split_indices=split_indices,
                **kwargs,
            )
        else:
            # Use all elements
            fp_new, g_new = self.modify_fp_elements(
                fp=fp_new,
                g=g_new,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                split_indices_nm=split_indices_nm,
                **kwargs,
            )
        # Add a small number to avoid division by zero
        fp_new += self.eps
        # Calculate the root of the sum
        if self.use_roots:
            if g is not None:
                mroot = (1.0 / power) - 1.0
                g_new = g_new * (fp_new**mroot)[..., None]
            root = 1.0 / power
            fp_new = fp_new**root
        else:
            if g is not None:
                g_new *= power
        return fp_new, g_new

    def update_arguments(
        self,
        reduce_dimensions=None,
        use_derivatives=None,
        wrap=None,
        include_ncells=None,
        periodic_sum=None,
        periodic_softmax=None,
        mic=None,
        all_ncells=None,
        cell_cutoff=None,
        use_cutoff=None,
        rs_cutoff=None,
        re_cutoff=None,
        dtype=None,
        use_tags=None,
        use_pairs=None,
        reuse_combinations=None,
        power=None,
        use_roots=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Calculate and store derivatives of the fingerprint wrt.
                the cartesian coordinates.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            include_ncells: bool
                Include the neighboring cells when calculating the distances.
                The fingerprint will include the neighboring cells.
                include_ncells will replace periodic_softmax and mic.
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
            periodic_sum: bool
                Use a sum of the distances to neighboring cells
                when periodic boundary conditions are used.
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
            periodic_softmax: bool
                Use a softmax weighting on the distances to neighboring cells
                from the squared distances when periodic boundary conditions
                are used.
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
                Either use mic, periodic_sum, periodic_softmax, or
                include_ncells.
                mic is faster than periodic_softmax,
                but the derivatives are discontinuous.
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
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
            use_tags: bool
                Use the tags of the atoms to identify the atoms as
                another type.
            use_pairs: bool
                Whether to use pairs of elements or use all elements.
            reuse_combinations: bool
                Whether to reuse the combinations of the elements.
                The change in the atomic numbers and tags will be checked
                to see if they are unchanged.
                If False, the combinations are calculated each time.
            power: int
                The power of the inverse distances.
            use_roots: bool
                Whether to use roots of the power elements.

        Returns:
            self: The updated instance itself.
        """
        super().update_arguments(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            wrap=wrap,
            include_ncells=include_ncells,
            periodic_sum=periodic_sum,
            periodic_softmax=periodic_softmax,
            mic=mic,
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            use_cutoff=use_cutoff,
            rs_cutoff=rs_cutoff,
            re_cutoff=re_cutoff,
            dtype=dtype,
            use_tags=use_tags,
            use_pairs=use_pairs,
            reuse_combinations=reuse_combinations,
        )
        if power is not None:
            self.power = int(power)
        if use_roots is not None:
            self.use_roots = use_roots
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            wrap=self.wrap,
            include_ncells=self.include_ncells,
            periodic_sum=self.periodic_sum,
            periodic_softmax=self.periodic_softmax,
            mic=self.mic,
            all_ncells=self.all_ncells,
            cell_cutoff=self.cell_cutoff,
            use_cutoff=self.use_cutoff,
            rs_cutoff=self.rs_cutoff,
            re_cutoff=self.re_cutoff,
            dtype=self.dtype,
            use_tags=self.use_tags,
            use_pairs=self.use_pairs,
            reuse_combinations=self.reuse_combinations,
            power=self.power,
            use_roots=self.use_roots,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
