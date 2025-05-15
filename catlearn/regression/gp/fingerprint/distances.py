from numpy import arange, asarray, full, repeat, sqrt, zeros
from .geometry import (
    check_atoms,
    get_all_distances,
    get_constraints,
    get_covalent_distances,
    get_mask_indicies,
    get_periodic_softmax,
    get_periodic_sum,
)
from .fingerprint import Fingerprint


class Distances(Fingerprint):
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
        dtype=float,
        **kwargs,
    ):
        """
        Fingerprint constructer class that convert atoms object into
        a fingerprint object with vector and derivatives.
        The distance fingerprint constructer class.
        The distances are scaled with covalent radii.

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
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set the arguments
        self.update_arguments(
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
            **kwargs,
        )

    def make_fingerprint(self, atoms, **kwargs):
        # Get the masked and not masked atoms
        not_masked, masked = get_constraints(
            atoms,
            reduce_dimensions=self.reduce_dimensions,
        )
        # Initialize the masking and indicies
        (
            not_masked,
            masked,
            nmi,
            nmj,
            nmi_ind,
            nmj_ind,
        ) = get_mask_indicies(atoms, not_masked=not_masked, masked=masked)
        # Get the periodicity
        pbc = atoms.pbc
        # Check what distance method should be used
        (
            use_vector,
            use_include_ncells,
            use_periodic_softmax,
            use_periodic_sum,
            use_mic,
        ) = self.use_dis_method(pbc=pbc, **kwargs)
        # Check whether to calculate neighboring cells
        use_ncells = (
            use_include_ncells or use_periodic_softmax or use_periodic_sum
        )
        # Get all the distances and their vectors
        dist, dist_vec = self.get_distances(
            atoms=atoms,
            not_masked=not_masked,
            masked=masked,
            nmi=nmi,
            nmj=nmj,
            nmi_ind=nmi_ind,
            nmj_ind=nmj_ind,
            use_vector=use_vector,
            include_ncells=use_ncells,
            mic=use_mic,
        )
        # Calculate the fingerprint and its derivatives
        fp, g = self.calc_fp(
            dist=dist,
            dist_vec=dist_vec,
            not_masked=not_masked,
            masked=masked,
            nmi=nmi,
            nmj=nmj,
            nmi_ind=nmi_ind,
            nmj_ind=nmj_ind,
            atomic_numbers=atoms.get_atomic_numbers(),
            tags=atoms.get_tags(),
            use_include_ncells=use_include_ncells,
            use_periodic_sum=use_periodic_sum,
            use_periodic_softmax=use_periodic_softmax,
        )
        return fp, g

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
        dist = sqrt(dist**2 + self.eps)
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
        fp = dist / covdis
        # Check what distance method should be used
        if use_periodic_softmax:
            # Calculate the fingerprint with the periodic softmax
            fp, g = get_periodic_softmax(
                dist_eps=dist,
                dist_vec=dist_vec,
                fpinner=fp,
                covdis=covdis,
                use_inv_dis=False,
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
                use_inv_dis=False,
                use_derivatives=self.use_derivatives,
                **kwargs,
            )
        else:
            # Get the derivative of the fingerprint
            if self.use_derivatives:
                g = dist_vec * (-fp / (dist**2))[..., None]
            else:
                g = None
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

    def insert_to_deriv_matrix(
        self,
        g,
        not_masked,
        masked,
        nmi,
        nmj,
        use_include_ncells=False,
        **kwargs,
    ):
        """
        Insert the distance vectors into the derivative matrix.
        """
        # Get the length of the distance vector parts
        len_nm_m, len_nm, _ = self.get_length_dist(
            not_masked,
            masked,
            nmi,
        )
        # Get the indices for the distances
        i_m = arange(len_nm_m)
        i_nm_r = len_nm_m // len(not_masked)
        i_nm = repeat(arange(len(not_masked)), i_nm_r)
        i_nm_nm = arange(len_nm) + len_nm_m
        # Check if neighboring cells should be used
        if use_include_ncells:
            # Get the number of neighboring cells
            c_dim = len(g)
            # Make the derivative matrix
            deriv_matrix = zeros(
                (c_dim, len(g[0]), len(not_masked), 3),
                dtype=self.dtype,
            )
        else:
            # Make the derivative matrix
            deriv_matrix = zeros(
                (len(g), len(not_masked), 3),
                dtype=self.dtype,
            )
        # Fill the derivative matrix for masked with not masked
        deriv_matrix[..., i_m, i_nm, :] = g[..., i_m, :]
        # Fill the derivative matrix for not masked with not masked
        g_nm = g[..., i_nm_nm, :]
        deriv_matrix[..., i_nm_nm, nmi, :] = g_nm
        deriv_matrix[..., i_nm_nm, nmj, :] = -g_nm
        # Reshape the derivative matrix
        deriv_matrix = deriv_matrix.reshape(-1, len(not_masked) * 3)
        return deriv_matrix

    def use_dis_method(self, pbc, **kwargs):
        """
        Check what distance method should be used."

        Parameters:
            pbc: bool
                The periodic boundary conditions.

        Returns:
            use_vector: bool
                Whether to use the vector of the distances.
            use_include_ncells: bool
                Whether to include the neighboring cells when calculating
                the distances.
            use_periodic_softmax: bool
                Whether to use the periodic softmax.
            use_periodic_sum: bool
                Whether to use the periodic sum.
            use_mic: bool
                Whether to use the minimum image convention.
        """
        if not pbc.any():
            return self.use_derivatives, False, False, False, False
        if self.include_ncells:
            return True, True, False, False, False
        if self.periodic_softmax:
            return True, False, True, False, False
        if self.periodic_sum:
            return True, False, False, True, False
        if self.mic:
            return True, False, False, False, True
        return self.use_derivatives, False, False, False, False

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
        # Reshape the fingerprint
        if use_include_ncells:
            fp = fp.reshape(-1)
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
        return fp, g

    def element_setup(
        self,
        atomic_numbers,
        tags,
        not_masked,
        masked,
        use_include_ncells=False,
        c_dim=None,
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
                    return self.split_indicies
        # Save the atomic setup
        self.atomic_numbers = atomic_numbers
        self.not_masked = not_masked
        self.tags = tags
        # Merge element type and their tags
        if not self.use_tags:
            tags = zeros((len(atomic_numbers)), dtype=int)
        if len(not_masked):
            combis_nm = list(zip(atomic_numbers[not_masked], tags[not_masked]))
        else:
            combis_nm = []
        if len(masked):
            combis_m = list(zip(atomic_numbers[masked], tags[masked]))
        else:
            combis_m = []
        split_indicies = {}
        t = 0
        for i, i_nm in enumerate(combis_nm):
            i1 = i + 1
            for j_m in combis_m:
                split_indicies.setdefault(i_nm + j_m, []).append(t)
                t += 1
            for j_nm in combis_nm[i1:]:
                split_indicies.setdefault(i_nm + j_nm, []).append(t)
                t += 1
        # Include the neighboring cells
        if use_include_ncells and c_dim is not None:
            n_combi = full((c_dim, 1), t, dtype=int)
            split_indicies = {
                k: (asarray(v) + n_combi).reshape(-1)
                for k, v in split_indicies.items()
            }
        else:
            split_indicies = {k: asarray(v) for k, v in split_indicies.items()}
        # Save the split indicies
        self.split_indicies = split_indicies
        return split_indicies

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
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated instance itself.
        """
        super().update_arguments(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            dtype=dtype,
        )
        if wrap is not None:
            self.wrap = wrap
        if include_ncells is not None:
            self.include_ncells = include_ncells
        if periodic_sum is not None:
            self.periodic_sum = periodic_sum
        if periodic_softmax is not None:
            self.periodic_softmax = periodic_softmax
        if mic is not None:
            self.mic = mic
        if all_ncells is not None:
            self.all_ncells = all_ncells
        if cell_cutoff is not None:
            self.cell_cutoff = abs(float(cell_cutoff))
        if not hasattr(self, "not_masked"):
            self.not_masked = None
        if not hasattr(self, "masked"):
            self.masked = None
        if not hasattr(self, "atomic_numbers"):
            self.atomic_numbers = None
        if not hasattr(self, "tags"):
            self.tags = None
        if not hasattr(self, "split_indicies"):
            self.split_indicies = None
        # Tags is not implemented
        self.use_tags = False
        self.reuse_combinations = False
        return self

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
        return get_all_distances(
            atoms=atoms,
            not_masked=not_masked,
            masked=masked,
            nmi=nmi,
            nmj=nmj,
            nmi_ind=nmi_ind,
            nmj_ind=nmj_ind,
            use_vector=use_vector,
            wrap=self.wrap,
            include_ncells=include_ncells,
            mic=mic,
            all_ncells=self.all_ncells,
            cell_cutoff=self.cell_cutoff,
            dtype=self.dtype,
            **kwargs,
        )

    def get_length_dist(self, not_masked, masked, nmi, **kwargs):
        "Get the length of the distance vector parts."
        # Get the length of the distance vector parts
        len_nm_m = len(not_masked) * len(masked)
        len_nm = len(nmi)
        # Get the full length of the distance vector
        len_all = len_nm_m + len_nm
        return len_nm_m, len_nm, len_all

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
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
