from numpy import argsort, concatenate
from .invdistances import InvDistances


class SortedInvDistances(InvDistances):
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
        use_sort_all=False,
        reuse_combinations=True,
        **kwargs,
    ):
        """
        Fingerprint constructer class that convert atoms object into
        a fingerprint object with vector and derivatives.
        The sorted inverse distance fingerprint constructer class.
        The inverse distances are scaled with covalent radii.

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
            use_sort_all: bool
                Whether sort all the the combinations independently of the
                pairs.
            reuse_combinations: bool
                Whether to reuse the combinations of the elements.
                The change in the atomic numbers and tags will be checked
                to see if they are unchanged.
                If False, the combinations are calculated each time.
        """
        # Set the arguments
        super().__init__(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            wrap=wrap,
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            include_ncells=include_ncells,
            periodic_sum=periodic_sum,
            periodic_softmax=periodic_softmax,
            mic=mic,
            use_cutoff=use_cutoff,
            rs_cutoff=rs_cutoff,
            re_cutoff=re_cutoff,
            dtype=dtype,
            use_tags=use_tags,
            use_sort_all=use_sort_all,
            reuse_combinations=reuse_combinations,
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
        use_include_ncells=False,
        **kwargs,
    ):
        "Modify the fingerprint."
        # Sort the fingerprint
        if self.use_sort_all:
            fp, indicies = self.sort_fp_all(
                fp,
                use_include_ncells=use_include_ncells,
                **kwargs,
            )
        else:
            fp, indicies = self.sort_fp_pair(
                fp,
                atomic_numbers,
                tags,
                not_masked,
                masked,
                use_include_ncells=use_include_ncells,
                **kwargs,
            )
        # Sort the fingerprints and their derivatives
        fp = fp[indicies]
        # Insert the derivatives into the derivative matrix
        if g is not None:
            g = self.insert_to_deriv_matrix(
                g=g,
                not_masked=not_masked,
                masked=masked,
                nmi=nmi,
                nmj=nmj,
                use_include_ncells=use_include_ncells,
            )
            g = g[indicies]
        return fp, g

    def sort_fp_all(self, fp, use_include_ncells=False, **kwargs):
        "Get the indicies for sorting the fingerprint."
        # Reshape the fingerprint
        if use_include_ncells:
            fp = fp.reshape(-1)
        # Get the sorted indicies
        indicies = argsort(fp)
        return fp, indicies

    def sort_fp_pair(
        self,
        fp,
        atomic_numbers,
        tags,
        not_masked,
        masked,
        use_include_ncells=False,
        **kwargs,
    ):
        "Get the indicies for sorting the fingerprint."
        # Get the indicies of the atomic combinations
        split_indicies = self.element_setup(
            atomic_numbers,
            tags,
            not_masked,
            masked,
            use_include_ncells=use_include_ncells,
            c_dim=len(fp),
            **kwargs,
        )
        # Reshape the fingerprint
        if use_include_ncells:
            fp = fp.reshape(-1)
        # Sort the indicies after inverse distance magnitude
        indicies = [
            indi[argsort(fp[indi])] for indi in split_indicies.values()
        ]
        indicies = concatenate(indicies)
        return fp, indicies

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
        use_sort_all=None,
        reuse_combinations=None,
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
            use_sort_all: bool
                Whether sort all the the combinations independently of the
                pairs.
            reuse_combinations: bool
                Whether to reuse the combinations of the elements.
                The change in the atomic numbers and tags will be checked
                to see if they are unchanged.
                If False, the combinations are calculated each time.

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
        )
        if use_tags is not None:
            self.use_tags = use_tags
        if use_sort_all is not None:
            self.use_sort_all = use_sort_all
        if reuse_combinations is not None:
            self.reuse_combinations = reuse_combinations
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            wrap=self.wrap,
            all_ncells=self.all_ncells,
            cell_cutoff=self.cell_cutoff,
            include_ncells=self.include_ncells,
            periodic_sum=self.periodic_sum,
            periodic_softmax=self.periodic_softmax,
            mic=self.mic,
            use_cutoff=self.use_cutoff,
            rs_cutoff=self.rs_cutoff,
            re_cutoff=self.re_cutoff,
            dtype=self.dtype,
            use_tags=self.use_tags,
            use_sort_all=self.use_sort_all,
            reuse_combinations=self.reuse_combinations,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
