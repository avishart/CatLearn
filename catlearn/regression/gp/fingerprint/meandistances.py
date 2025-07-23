from numpy import asarray, zeros
from .sumdistances import SumDistances


class MeanDistances(SumDistances):
    """
    Fingerprint constructor class that convert an atoms instance into
    a fingerprint instance with vector and derivatives.
    The mean of inverse distance fingerprint constructer class.
    The inverse distances are scaled with covalent radii.
    """

    def modify_fp_pairs(
        self,
        fp,
        g,
        not_masked,
        use_include_ncells,
        split_indices_nm,
        split_indices,
        **kwargs,
    ):
        # Mean the fingerprints and derivatives if neighboring cells are used
        if use_include_ncells:
            fp = fp.mean(axis=0)
            if g is not None:
                g = g.mean(axis=0)
        # Make the new fingerprint
        fp_new = zeros(
            (len(split_indices_nm), len(split_indices)),
            dtype=self.dtype,
        )
        # Calculate the new derivatives
        if g is not None:
            # Make the new derivatives
            g_new = zeros(
                (
                    len(split_indices_nm),
                    len(split_indices),
                    len(not_masked),
                    3,
                ),
                dtype=self.dtype,
            )
            # Mean the fingerprint and derivatives
            for i, i_v in enumerate(split_indices_nm.values()):
                fp_i = fp[i_v]
                g_i = g[i_v]
                g_ij = g_i[:, not_masked].sum(axis=0)
                for j, (comb, j_v) in enumerate(split_indices.items()):
                    fp_new[i, j] = fp_i[:, j_v].mean()
                    n_comb = len(i_v) * len(j_v)
                    g_new[i, j, i_v] = g_i[:, j_v].sum(axis=1) / n_comb
                    if comb in split_indices_nm:
                        ij_comb = split_indices_nm[comb]
                        g_new[i, j, ij_comb] -= g_ij[ij_comb] / n_comb
            return fp_new.reshape(-1), g_new.reshape(-1, len(not_masked) * 3)
        # Mean the fingerprints
        for i, i_v in enumerate(split_indices_nm.values()):
            fp_i = fp[i_v]
            for j, j_v in enumerate(split_indices.values()):
                fp_new[i, j] = fp_i[:, j_v].mean()
        return fp_new.reshape(-1), None

    def modify_fp_elements(
        self,
        fp,
        g,
        not_masked,
        use_include_ncells,
        split_indices_nm,
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
            [fp[i_v].mean() for i_v in split_indices_nm.values()],
            dtype=self.dtype,
        )
        # Calculate the new derivatives
        if g is not None:
            g_new = zeros((len(split_indices_nm), len(not_masked), 3))
            for i, i_v in enumerate(split_indices_nm.values()):
                g_new[i, i_v] = g[i_v].sum(axis=1)
                g_new[i] -= g[i_v][:, not_masked].sum(axis=0)
                g_new[i] /= len(i_v) * n_atoms
            g_new = g_new.reshape(-1, len(not_masked) * 3)
            return fp, g_new
        return fp, None
