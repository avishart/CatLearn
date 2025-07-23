from numpy import (
    arange,
    asarray,
    ceil,
    concatenate,
    cos,
    einsum,
    exp,
    matmul,
    pi,
    sin,
    sqrt,
    triu_indices,
    where,
)
from numpy.linalg import pinv
import itertools
from scipy.spatial.distance import cdist
from ase.data import covalent_radii
from ase.constraints import FixAtoms


def get_constraints(atoms, reduce_dimensions=True, **kwargs):
    """
    Get the indices of the atoms that does not have fixed constraints.

    Parameters:
        atoms: ASE Atoms
            The ASE Atoms instance.
        reduce_dimensions: bool
            Whether to fix or mask some of the atoms.

    Returns:
        not_masked: (Nnm) list
            A list of indices for the moving atoms if constraints are used.
        masked: (Nm) list
            A list of indices for the fixed atoms if constraints are used.

    """
    not_masked = list(range(len(atoms)))
    if reduce_dimensions and len(atoms.constraints):
        masked = [
            c.get_indices()
            for c in atoms.constraints
            if isinstance(c, FixAtoms)
        ]
        if len(masked):
            masked = set(concatenate(masked))
            not_masked = list(set(not_masked).difference(masked))
            not_masked = sorted(not_masked)
            masked = list(masked)
    else:
        masked = []
    return asarray(not_masked), asarray(masked)


def get_mask_indices(
    atoms,
    not_masked=None,
    masked=None,
    nmi=None,
    nmj=None,
    nmi_ind=None,
    nmj_ind=None,
    **kwargs,
):
    """
    Get the indices of the atoms that are masked and not masked.

    Parameters:
        atoms: ASE Atoms
            The ASE Atoms instance.
        not_masked: (Nnm) list (optional)
            A list of indices for the moving atoms if constraints are used.
            Else all atoms are treated to be moving.
        masked: (Nn) list (optional)
            A list of indices for the fixed atoms if constraints are used.
        nmi: list (optional)
            The upper triangle indices of the not masked atoms.
        nmj: list (optional)
            The upper triangle indices of the not masked atoms.
        nmi_ind: list (optional)
            The indices of the not masked atoms.
        nmj_ind: list (optional)
            The indices of the not masked atoms.

    Returns:
        not_masked: (Nnm) list
            A list of indices for the moving atoms if constraints are used.
        masked: (Nm) list
            A list of indices for the fixed atoms if constraints are used.
        nmi: list
            The upper triangle indices of the not masked atoms.
        nmi_ind: list
            The indices of the not masked atoms.
        nmj_ind: list
            The indices of the not masked atoms.
    """
    # If a not masked list is given, all atoms is treated to be not masked
    if not_masked is None:
        not_masked = arange(len(atoms))
    # If a masked list is not given, it is calculated from the not masked
    if masked is None:
        masked = asarray(
            list(set(range(len(atoms))).difference(set(not_masked)))
        )
    # Make indices of not masked atoms with itself
    if nmi is None or nmj is None or nmi_ind is None or nmj_ind is None:
        nmi, nmj = triu_indices(len(not_masked), k=1, m=None)
        nmi_ind = not_masked[nmi]
        nmj_ind = not_masked[nmj]
    return not_masked, masked, nmi, nmj, nmi_ind, nmj_ind


def check_atoms(
    atomic_numbers,
    atomic_numbers_test,
    tags=None,
    tags_test=None,
    cell=None,
    cell_test=None,
    pbc=None,
    pbc_test=None,
    not_masked=None,
    not_masked_test=None,
    **kwargs,
):
    """
    Check if the atoms instance is the same as the input.

    Parameters:
        atomic_numbers: (N) list
            The atomic numbers of the atoms.
        atomic_numbers_test: (N) list
            The atomic numbers of the tested atoms.
        tags: (N) list (optional)
            The tags of the atoms.
        tags_test: (N) list (optional)
            The tags of the tested atoms.
        cell: (3, 3) array (optional)
            The cell vectors.
        cell_test: (3, 3) array (optional)
            The cell vectors of the tested atoms.
        pbc: (3) list (optional)
            The periodic boundary conditions.
        pbc_test: (3) list (optional)
            The periodic boundary conditions of the tested atoms.
        not_masked: (Nnm) list (optional)
            A list of indices for the moving atoms if constraints are used.
        not_masked_test: (Nnm) list (optional)
            A list of indices for the moving atoms if constraints
            are used in the tested atoms.

    Returns:
        bool: If the atoms are the same.
    """
    if len(atomic_numbers_test) != len(atomic_numbers):
        return False
    if not_masked is not None and not_masked_test is not None:
        if (not_masked_test != not_masked).any():
            return False
    if (atomic_numbers_test != atomic_numbers).any():
        return False
    if tags is not None and tags_test is not None:
        if (tags_test != tags).any():
            return False
    if cell is not None and cell_test is not None:
        if (cell_test != cell).any():
            return False
    if pbc is not None and pbc_test is not None:
        if (pbc_test != pbc).any():
            return False
    return True


def get_ncells(
    cell,
    pbc,
    all_ncells=False,
    cell_cutoff=4.0,
    atomic_numbers=None,
    remove0=False,
    dtype=float,
    **kwargs,
):
    """
    Get all neighboring cells within the cutoff.

    Parameters:
        cell: (3, 3) array
            The cell vectors.
        pbc: (3) list
            The periodic boundary conditions.
        all_ncells: bool
            If all neighboring cells within a cutoff should be used.
        cell_cutoff: float
            The distance cutoff for neighboring cells.
        atomic_numbers: list
            The atomic numbers of the atoms.
            It is only used when all_ncells is True.
        remove0: bool
            If the zero vector should
            be removed from the neighboring cells.
        dtype: type
            The data type of the arrays

    Returns:
        cells_p: (Nc, 3) array
            The displacements from all combinations of the neighboring cells.
    """
    # Check if all neighboring cells should be used
    if all_ncells:
        # Get the inverse of the cell
        cinv = pinv(cell)
        # Get the maximum covalent distance
        atomic_numbers_set = list(set(atomic_numbers))
        covrad = covalent_radii[atomic_numbers_set]
        max_cov = 2.0 * covrad.max()
        # Get the cutoff distance from the maximum covalent distance
        cutoff = max_cov * cell_cutoff
        # Get the coordinates to cutoff in lattice coordinates
        ccut = cutoff * cinv
        # Get the number of neighboring cells in each direction
        ncells = ceil(abs(ccut).max(axis=0)).astype(int)
        # Only use neighboring cells if the dimension is periodic
        ncells = where(pbc, ncells, 0)
    else:
        # Only use neighboring cells if the dimension is periodic
        ncells = where(pbc, 1, 0)
    # Get all neighboring cells
    b = [list(range(-i, i + 1)) for i in ncells]
    # Make all periodic combinations
    p_arrays = list(itertools.product(*b))
    # Remove the initial combination
    p_arrays.remove((0, 0, 0))
    # Add the zero vector in the beginning
    if not remove0:
        p_arrays = [(0, 0, 0)] + p_arrays
    # Calculate all displacement vector from the cell vectors
    p_arrays = asarray(p_arrays, dtype=dtype)
    cells_p = matmul(p_arrays, cell, dtype=dtype)
    return cells_p


def get_full_distance_matrix(
    atoms,
    not_masked=None,
    use_vector=False,
    wrap=True,
    include_ncells=False,
    mic=False,
    all_ncells=False,
    cell_cutoff=4.0,
    dtype=float,
    **kwargs,
):
    """
    Get the full cartesian distance matrix between the atomes and including
    the vectors if requested.

    Parameters:
        atoms: ASE Atoms
            The ASE Atoms instance.
        not_masked: Nnm list (optional)
            A list of indices for the moving atoms if constraints are used.
            Else all atoms are treated to be moving.
        use_vector: bool
            If the distance vectors should be returned.
        wrap: bool
            If the atoms should be wrapped to the cell.
        include_ncells: bool
            If neighboring cells should be included.
        all_ncells: bool
            If all neighboring cells within a cutoff should be used.
        mic: bool
            If the minimum image convention should be used.
        cell_cutoff: float
            The distance cutoff for neighboring cells.
        dtype: type
            The data type of the arrays

    Returns:
        dist: (N, Nnm) or (Nc, N, Nnm) array
            The full distance matrix.
        dist_vec: (N, Nnm, 3) or (Nc, N, Nnm, 3) array
            The full distance matrix with directions if use_vector=True.
    """
    # If a not masked list is not given all atoms is treated to be not masked
    if not_masked is None:
        not_masked = arange(len(atoms))
    # Get the atomic positions
    pos = asarray(atoms.get_positions(wrap=wrap), dtype=dtype)
    # Get the periodic boundary conditions
    pbc = atoms.pbc.copy()
    is_pbc = pbc.any()
    # Check whether to calculate distance vectors
    if use_vector or (is_pbc and (include_ncells or mic)):
        # Get distance vectors
        dist_vec = pos - pos[not_masked, None]
    else:
        dist_vec = None
        # Return the distances
        D = cdist(pos[not_masked], pos)
        D = asarray(D, dtype=dtype)
        return D, None
    # Check if neighboring cells should be included
    if include_ncells and is_pbc:
        cells_p = get_ncells(
            cell=atoms.get_cell(),
            pbc=pbc,
            atomic_numbers=atoms.get_atomic_numbers(),
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            dtype=dtype,
        )
        # Calculate the distances to the atoms in all unit cell
        dist_vec = dist_vec + cells_p[:, None, None, :]
        dist = sqrt(einsum("ijlk,ijlk->ijl", dist_vec, dist_vec))
        return dist, dist_vec
    elif mic and is_pbc:
        # Get the distances with minimum image convention
        dist, dist_vec = mic_distance(
            dist_vec=dist_vec,
            cell=atoms.get_cell(),
            pbc=pbc,
            use_vector=use_vector,
            dtype=dtype,
            **kwargs,
        )
        return dist, dist_vec
    # Calculate the distances and return
    dist = sqrt(einsum("ijl,ijl->ij", dist_vec, dist_vec))
    return dist, dist_vec


def get_all_distances(
    atoms,
    not_masked=None,
    masked=None,
    nmi=None,
    nmj=None,
    nmi_ind=None,
    nmj_ind=None,
    use_vector=False,
    wrap=True,
    include_ncells=False,
    mic=False,
    all_ncells=False,
    cell_cutoff=4.0,
    dtype=float,
    **kwargs,
):
    """
    Get the unique cartesian distances between the atomes and including
    the vectors if use_vector=True.

    Parameters:
        atoms: ASE Atoms
            The ASE Atoms instance.
        not_masked: Nnm list (optional)
            A list of indices for the moving atoms if constraints are used.
            Else all atoms are treated to be moving.
        masked: Nm list (optional)
            A list of indices for the fixed atoms if constraints are used.
        nmi: list (optional)
            The upper triangle indices of the not masked atoms.
        nmi_ind: list (optional)
            The indices of the not masked atoms.
        nmj_ind: list (optional)
            The indices of the not masked atoms.
        use_vector: bool
            If the distance vectors should be returned.
        wrap: bool
            If the atoms should be wrapped to the cell.
        mic: bool
            If the minimum image convention should be used.
        include_ncells: bool
            If neighboring cells should be included.
        all_ncells: bool
            If all neighboring cells within a cutoff should be used.
        cell_cutoff: float
            The distance cutoff for neighboring cells.
        dtype: type
            The data type of the arrays

    Returns:
        dist: (Nnm*Nm+(Nnm*(Nnm-1)/2)) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2)) array
            The unique distances.
        dist_vec: (Nnm*Nm+(Nnm*(Nnm-1)/2), 3) or
            (Nc, Nnm*N+(Nnm*(Nnm-1)/2), 3) array
            The unique distances with directions if use_vector=True.
    """
    # Make indices
    not_masked, masked, nmi, _, nmi_ind, nmj_ind = get_mask_indices(
        atoms,
        not_masked=not_masked,
        masked=masked,
        nmi=nmi,
        nmj=nmj,
        nmi_ind=nmi_ind,
        nmj_ind=nmj_ind,
    )
    # Get the atomic positions
    pos = asarray(atoms.get_positions(wrap=wrap), dtype=dtype)
    # Get the periodic boundary conditions
    pbc = atoms.pbc.copy()
    is_pbc = pbc.any()
    # Check whether to calculate distance vectors
    if use_vector or (is_pbc and (include_ncells or mic)):
        # Get distance vectors
        dist_vec = get_distance_vectors(
            pos,
            not_masked,
            masked,
            nmi_ind,
            nmj_ind,
        )
    else:
        # Get the distances
        dist = get_distances(
            pos,
            not_masked,
            masked,
            nmi,
            nmj_ind,
        )
        return dist, None
    # Check if neighboring cells should be included
    if include_ncells and is_pbc:
        cells_p = get_ncells(
            cell=atoms.get_cell(),
            pbc=pbc,
            atomic_numbers=atoms.get_atomic_numbers(),
            all_ncells=all_ncells,
            cell_cutoff=cell_cutoff,
            dtype=dtype,
        )
        # Calculate the distances to the atoms in all unit cell
        dist_vec = dist_vec + cells_p[:, None, :]
        dist = sqrt(einsum("ijl,ijl->ij", dist_vec, dist_vec))
        return dist, dist_vec
    elif mic and is_pbc:
        # Get the distances with minimum image convention
        dist, dist_vec = mic_distance(
            dist_vec=dist_vec,
            cell=atoms.get_cell(),
            pbc=pbc,
            use_vector=use_vector,
            dtype=dtype,
            **kwargs,
        )
        return dist, dist_vec
    # Calculate the distances and return
    dist = sqrt(einsum("ij,ij->i", dist_vec, dist_vec))
    return dist, dist_vec


def get_distances(
    pos,
    not_masked,
    masked,
    nmi,
    nmj_ind,
    **kwargs,
):
    """
    Get the unique distances.

    Parameters:
        pos: (N, 3) array
            The atomic positions.
        not_masked: Nnm list
            A list of indices for the moving atoms if constraints are used.
        masked: Nm list
            A list of indices for the fixed atoms if constraints are used.
        nmi: list
            The upper triangle indices of the not masked atoms.
        nmj_ind: list
            The indices of the not masked atoms.

    Returns:
        dist: (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The unique distances.
    """
    # Get the distances matrix
    d = cdist(pos[not_masked], pos)
    d = asarray(d, dtype=pos.dtype)
    # Get the distances of the not masked atoms
    dist = d[nmi, nmj_ind]
    if len(masked):
        # Get the distances of the masked atoms
        dist = concatenate(
            [d[:, masked].reshape(-1), dist],
            axis=0,
        )
    return dist


def get_distance_vectors(
    pos,
    not_masked,
    masked,
    nmi_ind,
    nmj_ind,
    **kwargs,
):
    """
    Get the unique distance vectors.

    Parameters:
        pos: (N, 3) array
            The atomic positions.
        not_masked: Nnm list
            A list of indices for the moving atoms if constraints are used.
        masked: Nm list
            A list of indices for the fixed atoms if constraints are used.
        nmi_ind: list
            The indices of the not masked atoms.
        nmj_ind: list
            The indices of the not masked atoms.

    Returns:
        dist_vec: (Nnm*Nm+(Nnm*(Nnm-1)/2), 3) array
            The unique distance vectors.
    """
    # Calculate the distance vectors for the not masked atoms
    dist_vec = pos[nmj_ind] - pos[nmi_ind]
    # Check if masked atoms are used
    if len(masked):
        # Calculate the distance vectors for the masked atoms
        dist_vec = concatenate(
            [
                (pos[masked] - pos[not_masked, None]).reshape(-1, 3),
                dist_vec,
            ],
            axis=0,
        )
    return dist_vec


def get_covalent_distances(
    atomic_numbers,
    not_masked,
    masked,
    nmi_ind,
    nmj_ind,
    dtype=float,
    **kwargs,
):
    """
    Get the covalent distances.

    Parameters:
        atomic_numbers: (N) list
            The atomic numbers of the atoms.
        not_masked: Nnm list
            A list of indices for the moving atoms if constraints are used.
        masked: Nm list
            A list of indices for the fixed atoms if constraints are used.
        nmi_ind: list
            The indices of the not masked atoms.
        nmj_ind: list
            The indices of the not masked atoms.
        dtype: type
            The data type of the arrays.

    Returns:
        covdis: (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The covalent distances.
    """
    # Get the covalent radii
    covrad = asarray(covalent_radii[atomic_numbers], dtype=dtype)
    # Calculate the covalent distances for the not masked atoms
    covdis = covrad[nmj_ind] + covrad[nmi_ind]
    # Check if masked atoms are used
    if len(masked):
        # Calculate the covalent distances for the masked atoms
        covdis = concatenate(
            [
                (covrad[masked] + covrad[not_masked, None]).reshape(-1),
                covdis,
            ],
            axis=0,
        )
    return covdis


def mic_distance(dist_vec, cell, pbc, use_vector=False, dtype=float, **kwargs):
    """
    Get the minimum image convention of the distances.

    Parameters:
        dist_vec: (N, Nnm, 3) or (Nnm*Nm+(Nnm*(Nnm-1)/2) , 3) array
            The distance vectors.
        cell: (3, 3) array
            The cell vectors.
        pbc: (3) list
            The periodic boundary conditions.
        use_vector: bool
            If the distance vectors should be returned.
        dtype: type
            The data type of the arrays

    Returns:
        dist: (N, Nnm) or ((Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The shortest distances.
        dist_vec: (N, Nnm, 3) or ((Nnm*Nm+(Nnm*(Nnm-1)/2), 3) array
            The shortest distance vectors if requested.
    """
    # Get the squared cell vectors
    cell2 = cell**2
    # Save the shortest distances
    v2min = dist_vec**2
    if use_vector:
        vmin = dist_vec.copy()
    else:
        vmin = None
    # Find what dimensions have cubic unit cells and not
    d_c = []
    pbc_nc = [False, False, False]
    if pbc[0]:
        if cell2[0, 1] + cell2[0, 2] + cell2[1, 0] + cell2[2, 0] == 0.0:
            d_c.append(0)
        else:
            pbc_nc[0] = True
    if pbc[1]:
        if cell2[1, 0] + cell2[1, 2] + cell2[0, 1] + cell2[2, 1] == 0.0:
            d_c.append(1)
        else:
            pbc_nc[1] = True
    if pbc[2]:
        if cell2[2, 0] + cell2[2, 1] + cell2[0, 2] + cell2[1, 2] == 0.0:
            d_c.append(2)
        else:
            pbc_nc[2] = True
    # Check if the cell is cubic to do a simpler mic
    if len(d_c):
        v2min, vmin = mic_cubic_distance(
            dist_vec,
            v2min,
            vmin,
            d_c,
            cell,
            use_vector=use_vector,
            **kwargs,
        )
    else:
        v2min = v2min.sum(axis=-1)
    if sum(pbc_nc):
        # Do an extensive mic for the dimension that is not cubic
        v2min, vmin = mic_general_distance(
            dist_vec,
            v2min,
            vmin,
            cell,
            pbc_nc,
            use_vector=use_vector,
            dtype=dtype,
            **kwargs,
        )
    return sqrt(v2min), vmin


def mic_cubic_distance(
    dist_vec,
    v2min,
    vmin,
    d_c,
    cell,
    use_vector=False,
    **kwargs,
):
    """
    Get the minimum image convention of the distances for cubic unit cells.
    It is faster than the extensive mic.
    """
    # Iterate over the x-, y-, and z-dimensions if they are periodic and cubic
    for d in d_c:
        # Calculate the distances to the atoms in the next unit cell
        dv_new = dist_vec[..., d] + cell[d, d]
        dv2_new = dv_new**2
        # Save the new distances if they are shorter
        i = where(dv2_new < v2min[..., d])
        v2min[(*i, d)] = dv2_new[(*i,)]
        if use_vector:
            vmin[(*i, d)] = dv_new[(*i,)]
        # Calculate the distances to the atoms in the previous unit cell
        dv_new = dist_vec[..., d] - cell[d, d]
        dv2_new = dv_new**2
        # Save the new distances if they are shorter
        i = where(dv2_new < v2min[..., d])
        v2min[(*i, d)] = dv2_new[(*i,)]
        if use_vector:
            vmin[(*i, d)] = dv_new[(*i,)]
    # Calculate the distances
    v2min = v2min.sum(axis=-1)
    if use_vector:
        return v2min, vmin
    return v2min, None


def mic_general_distance(
    dist_vec,
    v2min,
    vmin,
    cell,
    pbc_nc,
    use_vector=False,
    dtype=float,
    **kwargs,
):
    """
    Get the minimum image convention of the distances for any unit cells with
    an extensive mic search.
    """
    # Calculate all displacement vectors from the cell vectors
    cells_p = get_ncells(
        cell=cell,
        pbc=pbc_nc,
        all_ncells=False,
        remove0=True,
        dtype=dtype,
    )
    # Iterate over all combinations
    for p_array in cells_p:
        # Calculate the distances to the atoms in the next unit cell
        dv_new = dist_vec + p_array
        D_new = (dv_new**2).sum(axis=-1)
        # Save the new distances if they are shorter
        i = where(D_new < v2min)
        v2min[(*i,)] = D_new[(*i,)]
        if use_vector:
            vmin[(*i,)] = dv_new[(*i,)]
    # Calculate the distances
    if use_vector:
        return v2min, vmin
    return v2min, None


def get_periodic_sum(
    dist_eps,
    dist_vec,
    fpinner,
    use_inv_dis=True,
    use_derivatives=True,
    **kwargs,
):
    """
    Get the periodic sum of the distances.

    Parameters:
        dist_eps: (Nc, N, Nnm) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2)) array
            The distances with a small number added.
        dist_vec: (Nc, N, Nnm, 3) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2), 3) array
            The distance vectors.
        fpinner: (Nc, N, Nnm) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2)) array
            The inner fingerprint.
        use_inv_dis: bool
            Whether the inverse distance is used.
        use_derivatives: bool
            If the derivatives of the fingerprint should be returned.

    Returns:
        fp: (N, Nnm) or (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The fingerprint.
        g: (N, Nnm, 3) or (Nnm*Nm+(Nnm*(Nnm-1)/2), 3) array
            The derivatives of the fingerprint if requested.
    """
    # Calculate the fingerprint
    fp = fpinner.sum(axis=0)
    # Calculate the derivatives of the fingerprint
    if use_derivatives:
        # Calculate the derivatives of the distances
        if use_inv_dis:
            inner_deriv = fpinner / (dist_eps**2)
        else:
            inner_deriv = -fpinner / (dist_eps**2)
        # Calculate the derivatives of the fingerprint
        g = einsum("c...d,c...->...d", dist_vec, inner_deriv)
    else:
        g = None
    return fp, g


def get_periodic_softmax(
    dist_eps,
    dist_vec,
    fpinner,
    covdis,
    use_inv_dis=True,
    use_derivatives=True,
    eps=1e-16,
    **kwargs,
):
    """
    Get the periodic softmax of the distances.

    Parameters:
        dist_eps: (Nc, N, Nnm) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2)) array
            The distances with a small number added.
        dist_vec: (Nc, N, Nnm, 3) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2), 3) array
            The distance vectors.
        fpinner: (Nc, N, Nnm) or (Nc, Nnm*N+(Nnm*(Nnm-1)/2)) array
            The inner fingerprint.
            If use_inv_dis is True, the fingerprint is the covalent distances
            divided by distances.
            Else, the fingerprint is distances divided by the covalent
            distances.
        covdis: (N, Nnm) or (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The covalent distances.
        use_inv_dis: bool
            Whether the inverse distance is used.
        use_derivatives: bool
            If the derivatives of the fingerprint should be returned.
        eps: float
            A small number to avoid division by zero.

    Returns:
        fp: (N, Nnm) or (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The fingerprint.
        g: (N, Nnm, 3) or (Nnm*Nm+(Nnm*(Nnm-1)/2), 3) array
            The derivatives of the fingerprint if requested.
    """
    # Calculate weights
    if use_inv_dis:
        w = exp(-((1.0 / fpinner) ** 2))
    else:
        w = exp(-(fpinner**2))
    w = w / (w.sum(axis=0) + eps)
    # Calculate the all the fingerprint elements with their weights
    fp_w = fpinner * w
    # Calculate the fingerprint
    fp = fp_w.sum(axis=0)
    # Calculate the derivatives of the fingerprint
    if use_derivatives:
        # Calculate the derivatives of the distances
        if use_inv_dis:
            inner_deriv = 1.0 / (dist_eps**2)
        else:
            inner_deriv = -1.0 / (dist_eps**2)
        # Calculate the derivatives of the weights
        inner_deriv += (2.0 / (covdis**2)) * (1.0 - (fp / fpinner))
        # Calculate the inner derivative
        inner_deriv = fp_w * inner_deriv
        # Calculate the derivatives of the fingerprint
        g = einsum("c...d,c...->...d", dist_vec, inner_deriv)
    else:
        g = None
    return fp, g


def cosine_cutoff(fp, g, rs_cutoff=3.0, re_cutoff=4.0, eps=1e-16, **kwargs):
    """
    Cosine cutoff function.
    Modification of eq. 24 in https://doi.org/10.1002/qua.24927.
    A small value has been added to the inverse distance to avoid division
    by zero.

    Parameters:
        fp: (N, Nnm) or (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The fingerprint.
        g: (N, Nnm, 3) or (Nnm*Nm+(Nnm*(Nnm-1)/2), 3) array
            The derivatives of the fingerprint.
        rs_cutoff: float
            The start of the cutoff function.
        re_cutoff: float
            The end of the cutoff function.
        eps: float
            A small number to avoid division by zero.

    Returns:
        fp: (N, Nnm) or (Nnm*Nm+(Nnm*(Nnm-1)/2)) array
            The fingerprint.
        g: (N, Nnm, 3) or (Nnm*Nm+(Nnm*(Nnm-1)/2), 3) array
            The derivatives of the fingerprint.
    """
    # Find the scale of the cutoff function
    rscale = re_cutoff - rs_cutoff
    # Calculate the inverse fingerprint with small number added
    fp_inv = 1.0 / (fp + eps)
    # Calculate the cutoff function
    fc_inner = pi * (fp_inv - rs_cutoff) / rscale
    fc = 0.5 * (cos(fc_inner) + 1.0)
    # Crop the cutoff function
    fp_rs = fp_inv < rs_cutoff
    fp_re = fp_inv > re_cutoff
    fc = where(fp_rs, 1.0, fc)
    fc = where(fp_re, 0.0, fc)
    # Calculate the derivative of the cutoff function
    if g is not None:
        gc = (0.5 * pi / rscale) * sin(fc_inner) * (fp_inv**2)
        gc = where(fp_rs, 0.0, gc)
        gc = where(fp_re, 0.0, gc)
        g = g * (fc + fp * gc)[..., None]
    # Multiply the fingerprint with the cutoff function
    fp = fp * fc
    return fp, g
