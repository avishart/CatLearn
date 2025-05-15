from .geometry import (
    cosine_cutoff,
    get_covalent_distances,
    get_periodic_softmax,
    get_periodic_sum,
)
from .distances import Distances


class InvDistances(Distances):
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
        The inverse distance fingerprint constructer class.
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
            dtype=dtype,
        )
        if use_cutoff is not None:
            self.use_cutoff = use_cutoff
        if rs_cutoff is not None:
            self.rs_cutoff = abs(float(rs_cutoff))
        if re_cutoff is not None:
            self.re_cutoff = abs(float(re_cutoff))
        return self

    def calc_fp(
        self,
        dist,
        dist_vec,
        not_masked,
        masked,
        nmi,
        nmj,
        nmi_ind,
        nmj_ind,
        atomic_numbers,
        tags=None,
        use_include_ncells=False,
        use_periodic_sum=False,
        use_periodic_softmax=False,
        **kwargs,
    ):
        "Calculate the fingerprint."
        # Add small number to avoid division by zero to the distances
        dist += self.eps
        # Get the covalent distances
        covdis = get_covalent_distances(
            atomic_numbers=atomic_numbers,
            not_masked=not_masked,
            masked=masked,
            nmi_ind=nmi_ind,
            nmj_ind=nmj_ind,
            dtype=self.dtype,
        )
        # Set the correct shape of the covalent distances
        if use_include_ncells or use_periodic_sum or use_periodic_softmax:
            covdis = covdis[None, ...]
        # Calculate the fingerprint
        fp = covdis / dist
        # Check what distance method should be used
        if use_periodic_softmax:
            # Calculate the fingerprint with the periodic softmax
            fp, g = get_periodic_softmax(
                dist_eps=dist,
                dist_vec=dist_vec,
                fpinner=fp,
                covdis=covdis,
                use_inv_dis=True,
                use_derivatives=self.use_derivatives,
                eps=self.eps,
                **kwargs,
            )
        elif use_periodic_sum:
            # Calculate the fingerprint with the periodic sum
            fp, g = get_periodic_sum(
                dist_eps=dist,
                dist_vec=dist_vec,
                fpinner=fp,
                use_inv_dis=True,
                use_derivatives=self.use_derivatives,
                **kwargs,
            )
        else:
            # Get the derivative of the fingerprint
            if self.use_derivatives:
                g = dist_vec * (fp / (dist**2))[..., None]
            else:
                g = None
        # Apply the cutoff function
        if self.use_cutoff:
            fp, g = self.apply_cutoff(fp, g, **kwargs)
        # Update the fingerprint with the modification
        fp, g = self.modify_fp(
            fp=fp,
            g=g,
            atomic_numbers=atomic_numbers,
            tags=tags,
            not_masked=not_masked,
            masked=masked,
            nmi=nmi,
            nmj=nmj,
            nmi_ind=nmi_ind,
            nmj_ind=nmj_ind,
            use_include_ncells=use_include_ncells,
            **kwargs,
        )
        return fp, g

    def apply_cutoff(self, fp, g, **kwargs):
        "Get the cutoff function."
        return cosine_cutoff(
            fp,
            g,
            rs_cutoff=self.rs_cutoff,
            re_cutoff=self.re_cutoff,
            eps=self.eps,
            **kwargs,
        )

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
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
