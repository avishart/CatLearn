from numpy import asarray, zeros
from .sumdistances import SumDistances


class MeanDistances(SumDistances):
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
        The mean of inverse distance fingerprint constructer class.
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
                Use the pairs of the atoms to identify the atoms as
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
        # Mean the fingerprints and derivatives if neighboring cells are used
        if use_include_ncells:
            fp = fp.mean(axis=0)
            if g is not None:
                g = g.mean(axis=0)
        # Make the new fingerprint
        fp_new = zeros(
            (len(split_indicies_nm), len(split_indicies)),
            dtype=self.dtype,
        )
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
            # Mean the fingerprint and derivatives
            for i, i_v in enumerate(split_indicies_nm.values()):
                fp_i = fp[i_v]
                g_i = g[i_v]
                g_ij = g_i[:, not_masked].sum(axis=0)
                for j, (comb, j_v) in enumerate(split_indicies.items()):
                    fp_new[i, j] = fp_i[:, j_v].mean()
                    n_comb = len(i_v) * len(j_v)
                    g_new[i, j, i_v] = g_i[:, j_v].sum(axis=1) / n_comb
                    if comb in split_indicies_nm:
                        ij_comb = split_indicies_nm[comb]
                        g_new[i, j, ij_comb] -= g_ij[ij_comb] / n_comb
            return fp_new.reshape(-1), g_new.reshape(-1, len(not_masked) * 3)
        # Mean the fingerprints
        for i, i_v in enumerate(split_indicies_nm.values()):
            fp_i = fp[i_v]
            for j, j_v in enumerate(split_indicies.values()):
                fp_new[i, j] = fp_i[:, j_v].mean()
        return fp_new.reshape(-1), None

    def modify_fp_elements(
        self,
        fp,
        g,
        not_masked,
        use_include_ncells,
        split_indicies_nm,
        **kwargs,
    ):
        # Mean the fingerprints and derivatives if neighboring cells are used
        if use_include_ncells:
            fp = fp.mean(axis=0)
            if g is not None:
                g = g.mean(axis=0)
        # Mean the fingerprints
        n_atoms = fp.shape[1]
        fp = fp.mean(axis=1)
        fp = asarray(
            [fp[i_v].mean() for i_v in split_indicies_nm.values()],
            dtype=self.dtype,
        )
        # Calculate the new derivatives
        if g is not None:
            g_new = zeros((len(split_indicies_nm), len(not_masked), 3))
            for i, i_v in enumerate(split_indicies_nm.values()):
                g_new[i, i_v] = g[i_v].sum(axis=1)
                g_new[i] -= g[i_v][:, not_masked].sum(axis=0)
                g_new[i] /= len(i_v) * n_atoms
            g_new = g_new.reshape(-1, len(not_masked) * 3)
            return fp, g_new
        return fp, None
