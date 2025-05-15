from numpy import arange, asarray, zeros
from ase.data import covalent_radii
from .invdistances import InvDistances
from ..fingerprint.geometry import (
    check_atoms,
    get_full_distance_matrix,
    get_periodic_softmax,
    get_periodic_sum,
)


class SumDistances(InvDistances):
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
        **kwargs,
    ):
        """
        Fingerprint constructer class that convert atoms object into
        a fingerprint object with vector and derivatives.
        The sum of inverse distance fingerprint constructer class.
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
            use_pairs: bool
                Whether to use pairs of elements or use all elements.
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
        "Modify the fingerprint."
        # Get the indicies of the atomic combinations
        split_indicies_nm, split_indicies = self.element_setup(
            atomic_numbers,
            tags,
            not_masked,
            **kwargs,
        )
        # Modify the fingerprint
        if self.use_pairs:
            # Use pairs of elements
            fp, g = self.modify_fp_pairs(
                fp=fp,
                g=g,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                split_indicies_nm=split_indicies_nm,
                split_indicies=split_indicies,
                **kwargs,
            )
        else:
            # Use all elements
            fp, g = self.modify_fp_elements(
                fp=fp,
                g=g,
                not_masked=not_masked,
                use_include_ncells=use_include_ncells,
                split_indicies_nm=split_indicies_nm,
                **kwargs,
            )
        return fp, g

    def modify_fp_pairs(
        self,
        fp,
        g,
        not_masked,
        use_include_ncells,
        split_indicies_nm,
        split_indicies,
        **kwargs,
    ):
        "Modify the fingerprint over pairs of elements."
        # Sum the fingerprints and derivatives if neighboring cells are used
        if use_include_ncells:
            fp = fp.sum(axis=0)
            if g is not None:
                g = g.sum(axis=0)
        # Make the new fingerprint
        fp_new = zeros(
            (len(split_indicies_nm), len(split_indicies)),
            dtype=self.dtype,
        )
        # Sum the fingerprints
        for i, i_v in enumerate(split_indicies_nm.values()):
            fp_i = fp[i_v]
            for j, j_v in enumerate(split_indicies.values()):
                fp_new[i, j] = fp_i[:, j_v].sum()
        fp_new = fp_new.reshape(-1)
        # Calculate the new derivatives
        if g is not None:
            # Make the new derivatives
            g_new = zeros(
                (
                    len(split_indicies_nm),
                    len(split_indicies),
                    len(not_masked),
                    3,
                ),
                dtype=self.dtype,
            )
            # Sum the derivatives
            for i, i_v in enumerate(split_indicies_nm.values()):
                g_i = g[i_v]
                g_ij = g_i[:, not_masked].sum(axis=0)
                for j, (comb, j_v) in enumerate(split_indicies.items()):
                    g_new[i, j, i_v] = g_i[:, j_v].sum(axis=1)
                    if comb in split_indicies_nm:
                        ij_comb = split_indicies_nm[comb]
                        g_new[i, j, ij_comb] -= g_ij[ij_comb]
            g_new = g_new.reshape(-1, len(not_masked) * 3)
            return fp_new, g_new
        return fp_new, None

    def modify_fp_elements(
        self,
        fp,
        g,
        not_masked,
        use_include_ncells,
        split_indicies_nm,
        **kwargs,
    ):
        "Modify the fingerprint over all elements."
        # Sum the fingerprints and derivatives if neighboring cells are used
        if use_include_ncells:
            fp = fp.sum(axis=0)
            if g is not None:
                g = g.sum(axis=0)
        # Sum the fingerprints
        fp = fp.sum(axis=1)
        fp = asarray(
            [fp[i_v].sum() for i_v in split_indicies_nm.values()],
            dtype=self.dtype,
        )
        # Calculate the new derivatives
        if g is not None:
            g_new = zeros((len(split_indicies_nm), len(not_masked), 3))
            for i, i_v in enumerate(split_indicies_nm.values()):
                g_new[i, i_v] = g[i_v].sum(axis=1)
                g_new[i] -= g[i_v][:, not_masked].sum(axis=0)
            g_new = g_new.reshape(-1, len(not_masked) * 3)
            return fp, g_new
        return fp, None

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
        if use_pairs is not None:
            self.use_pairs = use_pairs
        if reuse_combinations is not None:
            self.reuse_combinations = reuse_combinations
        if not hasattr(self, "split_indicies_nm"):
            self.split_indicies_nm = None
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
        covdis = self.get_covalent_distances(
            atomic_numbers=atomic_numbers,
            not_masked=not_masked,
        )
        # Check if the distances include the neighboring cells
        if use_include_ncells or use_periodic_sum or use_periodic_softmax:
            covdis = covdis[None, ...]
        # Get the index of the not masked atoms
        i_nm = arange(len(not_masked))
        # Calculate the inverse distances
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
        # Remove self interaction
        fp[..., i_nm, not_masked] = 0.0
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

    def get_distances(
        self,
        atoms,
        not_masked=None,
        masked=None,
        nmi=None,
        nmj=None,
        nmi_ind=None,
        nmj_ind=None,
        use_vector=False,
        include_ncells=False,
        mic=False,
        **kwargs,
    ):
        """
        Get the distances and their vectors.
        """
        return get_full_distance_matrix(
            atoms=atoms,
            not_masked=not_masked,
            use_vector=use_vector,
            wrap=self.wrap,
            include_ncells=include_ncells,
            mic=mic,
            all_ncells=self.all_ncells,
            cell_cutoff=self.cell_cutoff,
            dtype=self.dtype,
        )

    def get_covalent_distances(self, atomic_numbers, not_masked):
        "Get the covalent distances of the atoms."
        cov_dis = covalent_radii[atomic_numbers]
        return asarray(cov_dis + cov_dis[not_masked, None], dtype=self.dtype)

    def element_setup(
        self,
        atomic_numbers,
        tags,
        not_masked,
        **kwargs,
    ):
        """
        Get all informations of the atom combinations and split them
        into types.
        """
        # Check if the atomic setup is the same
        if self.reuse_combinations:
            if (
                self.atomic_numbers is not None
                or self.not_masked is not None
                or self.tags is not None
                or self.split_indicies is not None
                or self.split_indicies_nm is not None
            ):
                atoms_equal = check_atoms(
                    atomic_numbers=self.atomic_numbers,
                    atomic_numbers_test=atomic_numbers,
                    tags=self.tags,
                    tags_test=tags,
                    not_masked=self.not_masked,
                    not_masked_test=not_masked,
                    **kwargs,
                )
                if atoms_equal:
                    return self.split_indicies_nm, self.split_indicies
        # Save the atomic numbers and tags
        self.atomic_numbers = atomic_numbers
        self.tags = tags
        self.not_masked = not_masked
        # Get the atomic types of the atoms
        if not self.use_tags:
            tags = zeros((len(atomic_numbers)), dtype=int)
        combis = list(zip(atomic_numbers, tags))
        split_indicies = {}
        for i, combi in enumerate(combis):
            split_indicies.setdefault(combi, []).append(i)
        self.split_indicies = split_indicies
        # Get the atomic types of the not masked atoms
        combis = list(zip(atomic_numbers[not_masked], tags[not_masked]))
        split_indicies_nm = {}
        for i, combi in enumerate(combis):
            split_indicies_nm.setdefault(combi, []).append(i)
        self.split_indicies_nm = split_indicies_nm
        return split_indicies_nm, split_indicies

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
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
