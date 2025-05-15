from .invdistances import InvDistances


class InvDistances2(InvDistances):
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
        **kwargs,
    ):
        """
        Fingerprint constructer class that convert atoms object into
        a fingerprint object with vector and derivatives.
        The inverse squared distance fingerprint constructer class.
        The inverse squared distances are scaled with covalent radii.

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
        # Adjust the derivatives so they are squared
        if g is not None:
            g = (2.0 * fp)[..., None] * g
            g = self.insert_to_deriv_matrix(
                g=g,
                not_masked=not_masked,
                masked=masked,
                nmi=nmi,
                nmj=nmj,
                use_include_ncells=use_include_ncells,
            )
        # Reshape the fingerprint
        if use_include_ncells:
            fp = fp.reshape(-1)
        # Adjust the fingerprint so it is squared
        fp = fp**2
        return fp, g
