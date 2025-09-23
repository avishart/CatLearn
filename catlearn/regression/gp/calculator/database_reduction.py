from numpy import (
    append,
    arange,
    argmax,
    argmin,
    argsort,
    array,
    asarray,
    delete,
    einsum,
    nanmin,
    sqrt,
)
from scipy.spatial.distance import cdist
from .database import Database


class DatabaseReduction(Database):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduction is done with a method that is defined in the class.
    """

    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        round_targets=None,
        seed=None,
        dtype=float,
        npoints=25,
        initial_indices=[0],
        include_last=1,
        **kwargs,
    ):
        """
        Initialize the database.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
        """
        # The negative forces have to be used since the derivatives are used
        self.use_negative_forces = True
        # Set initial indices
        self.indices = []
        # Use default fingerprint if it is not given
        if fingerprint is None:
            self.set_default_fp(
                reduce_dimensions=reduce_dimensions,
                use_derivatives=use_derivatives,
                dtype=dtype,
            )
        # Set the arguments
        self.update_arguments(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
            **kwargs,
        )

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
        round_targets=None,
        seed=None,
        dtype=None,
        npoints=None,
        initial_indices=None,
        include_last=None,
        **kwargs,
    ):
        """
        Update the class with its arguments. The existing arguments are used
        if they are not given.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.

        Returns:
            self: The updated object itself.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
        )
        # Set the number of points to use
        if npoints is not None:
            self.npoints = int(npoints)
        # Set the initial indices to keep fixed
        if initial_indices is not None:
            self.initial_indices = array(initial_indices, dtype=int)
        # Set the number of last points to include
        if include_last is not None:
            self.include_last = int(abs(include_last))
        # Check that too many last points are not included
        n_extra = self.npoints - len(self.initial_indices)
        if self.include_last > n_extra:
            self.include_last = n_extra if n_extra >= 0 else 0
        # Store that the data base has changed
        self.update_indices = True
        return self

    def get_all_data_atoms(self, **kwargs):
        """
        Get the list of all atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return super().get_data_atoms(**kwargs)

    def get_data_atoms(self, **kwargs):
        """
        Get the list of atoms in the reduced database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        indices = self.get_reduction_indices()
        atoms_list = self.get_all_data_atoms(**kwargs)
        return [atoms_list[i] for i in indices]

    def get_features(self, **kwargs):
        """
        Get the fingerprints of the atoms in the reduced database.

        Returns:
            array: A matrix array with the saved features or fingerprints.
        """
        indices = self.get_reduction_indices()
        if self.use_fingerprint:
            return array(self.features)[indices]
        return array(self.features, dtype=self.dtype)[indices]

    def get_all_feature_vectors(self, **kwargs):
        "Get all the features in numpy array form."
        if self.use_fingerprint:
            features = [feature.get_vector() for feature in self.features]
            return array(features, dtype=self.dtype)
        return array(self.features, dtype=self.dtype)

    def get_targets(self, **kwargs):
        """
        Get the targets of the atoms in the reduced database.

        Returns:
            array: A matrix array with the saved targets.
        """
        indices = self.get_reduction_indices()
        return array(self.targets, dtype=self.dtype)[indices]

    def get_all_targets(self, **kwargs):
        """
        Get all the targets of the atoms in the database.

        Returns:
            array: A matrix array with the saved targets.
        """
        return array(self.targets, dtype=self.dtype)

    def get_initial_indices(self, **kwargs):
        """
        Get the initial indices of the used atoms in the database.

        Returns:
            array: The initial indices of the atoms used.
        """
        return array(self.initial_indices, dtype=int)

    def get_last_indices(self, indices, not_indices, **kwargs):
        """
        Include the last indices that are not in the used indices list.

        Parameters:
            indices: list
                A list of used indices.
            not_indices: list
                A list of indices that not used yet.

        Returns:
            list: A list of the used indices including the last indices.
        """
        if self.include_last != 0:
            last = -self.include_last
            indices = append(
                indices,
                [not_indices[last:]],
            )
        return indices

    def get_not_indices(self, indices, all_indices, **kwargs):
        """
        Get a list of the indices that are not in the used indices list.

        Parameters:
            indices: list
                A list of indices.
            all_indices: list
                A list of all indices.

        Returns:
            list: A list of indices that not used.
        """
        return list(set(all_indices).difference(indices))

    def append(self, atoms, **kwargs):
        "Append the atoms object, the fingerprint, and target(s) to lists."
        # Store that the data base has changed
        self.update_indices = True
        # Append to the data base
        super().append(atoms, **kwargs)
        return self

    def get_reduction_indices(self, **kwargs):
        "Get the indices of the reduced data used."
        # If the indices is already calculated then give them
        if not self.update_indices:
            return self.indices
        # Set up all the indices
        self.update_indices = False
        data_len = self.__len__()
        all_indices = arange(data_len)
        # No reduction is needed if the database is not large
        if data_len <= self.npoints:
            self.indices = all_indices.copy()
            return self.indices
        # Reduce the data base
        self.indices = self.make_reduction(all_indices)
        return self.indices

    def make_reduction(self, all_indices, **kwargs):
        "Make the reduction of the data base with a chosen method."
        raise NotImplementedError()

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            round_targets=self.round_targets,
            seed=self.seed,
            dtype=self.dtype,
            npoints=self.npoints,
            initial_indices=self.initial_indices,
            include_last=self.include_last,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indices=self.update_indices)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indices=self.indices.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabaseDistance(DatabaseReduction):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduction is done by selecting the points with the
    largest distances from each other.
    """

    def make_reduction(self, all_indices, **kwargs):
        "Reduce the training set with the points farthest from each other."
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the last point
        indices = self.get_last_indices(indices, not_indices)
        # Get a random index if no fixed index exist
        if len(indices) == 0:
            indices = asarray([self.rng.choice(not_indices)], dtype=int)
            not_indices = self.get_not_indices(indices, all_indices)
        # Get all the features
        features = self.get_all_feature_vectors()
        fdim = len(features[0])
        for i in range(len(indices), self.npoints):
            # Get the indices for the system not already included
            not_indices = self.get_not_indices(indices, all_indices)
            # Calculate the distances to the points already used
            dist = cdist(
                features[indices].reshape(-1, fdim),
                features[not_indices].reshape(-1, fdim),
            )
            # Choose the point furthest from the points already used
            i_max = argmax(nanmin(dist, axis=0))
            indices = append(indices, [not_indices[i_max]])
        return array(indices, dtype=int)


class DatabaseRandom(DatabaseReduction):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduction is done by selecting the points randomly.
    """

    def make_reduction(self, all_indices, **kwargs):
        "Random select the training points."
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the last point
        indices = self.get_last_indices(indices, not_indices)
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Get the number of missing points
        npoints = int(self.npoints - len(indices))
        # Randomly get the indices
        indices = append(
            indices,
            self.rng.permutation(not_indices)[:npoints],
        )
        return array(indices, dtype=int)


class DatabaseHybrid(DatabaseReduction):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduction is done by selecting the points with the
    largest distances from each other and randomly.
    The random points are selected at every random_fraction step.
    """

    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        round_targets=None,
        seed=None,
        dtype=float,
        npoints=25,
        initial_indices=[0],
        include_last=1,
        random_fraction=3,
        **kwargs,
    ):
        """
        Initialize the database.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
            random_fraction: int
                How often the data point is sampled randomly.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
            random_fraction=random_fraction,
            **kwargs,
        )

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
        round_targets=None,
        seed=None,
        dtype=None,
        npoints=None,
        initial_indices=None,
        include_last=None,
        random_fraction=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
            random_fraction: int
                How often the data point is sampled randomly.

        Returns:
            self: The updated object itself.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
        )
        # Set the random fraction
        if random_fraction is not None:
            self.random_fraction = int(abs(random_fraction))
            if self.random_fraction == 0:
                self.random_fraction = 1
        return self

    def make_reduction(self, all_indices, **kwargs):
        """
        Use a combination of random sampling and
        farthest distance to reduce training set.
        """
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the last point
        indices = self.get_last_indices(indices, not_indices)
        # Get a random index if no fixed index exist
        if len(indices) == 0:
            indices = asarray([self.rng.choice(not_indices)], dtype=int)
            not_indices = self.get_not_indices(indices, all_indices)
        # Get all the features
        features = self.get_all_feature_vectors()
        fdim = len(features[0])
        for i in range(len(indices), self.npoints):
            # Get the indices for the system not already included
            not_indices = self.get_not_indices(indices, all_indices)
            if i % self.random_fraction == 0:
                # Get a random index
                indices = append(indices, [self.rng.choice(not_indices)])
            else:
                # Calculate the distances to the points already used
                dist = cdist(
                    features[indices].reshape(-1, fdim),
                    features[not_indices].reshape(-1, fdim),
                )
                # Choose the point furthest from the points already used
                i_max = argmax(nanmin(dist, axis=0))
                indices = append(indices, [not_indices[i_max]])
        return array(indices, dtype=int)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            round_targets=self.round_targets,
            seed=self.seed,
            dtype=self.dtype,
            npoints=self.npoints,
            initial_indices=self.initial_indices,
            include_last=self.include_last,
            random_fraction=self.random_fraction,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indices=self.update_indices)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indices=self.indices.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabaseMin(DatabaseReduction):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduction is done by selecting the points with the
    smallest targets.
    """

    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        round_targets=None,
        seed=None,
        dtype=float,
        npoints=25,
        initial_indices=[0],
        include_last=1,
        force_targets=False,
        **kwargs,
    ):
        """
        Initialize the database.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
            force_targets: bool
                Whether to include the derivatives/forces in targets
                when the smallest targets are found.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
            force_targets=force_targets,
            **kwargs,
        )

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
        round_targets=None,
        seed=None,
        dtype=None,
        npoints=None,
        initial_indices=None,
        include_last=None,
        force_targets=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
            force_targets: bool
                Whether to include the derivatives/forces in targets
                when the smallest targets are found.

        Returns:
            self: The updated object itself.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
        )
        # Set the force targets
        if force_targets is not None:
            self.force_targets = force_targets
        return self

    def make_reduction(self, all_indices, **kwargs):
        "Use the targets with smallest norms in the training set."
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the last point
        indices = self.get_last_indices(indices, not_indices)
        # Get the indices for the system not already included
        not_indices = array(self.get_not_indices(indices, all_indices))
        # Get the targets
        targets = self.get_all_targets()[not_indices]
        # Get sorting of the targets
        if self.force_targets:
            # Get the points with the lowest norm of the targets
            targets_norm = sqrt(einsum("ij,ij->i", targets, targets))
            i_sort = argsort(targets_norm)
        else:
            # Get the points with the lowest energies
            i_sort = argsort(targets[:, 0])
        # Get the number of missing points
        npoints = int(self.npoints - len(indices))
        # Get the indices for the system not already included
        i_sort = i_sort[:npoints]
        indices = append(indices, not_indices[i_sort])
        return array(indices, dtype=int)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            round_targets=self.round_targets,
            seed=self.seed,
            dtype=self.dtype,
            npoints=self.npoints,
            initial_indices=self.initial_indices,
            include_last=self.include_last,
            force_targets=self.force_targets,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indices=self.update_indices)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indices=self.indices.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabaseLast(DatabaseReduction):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduction is done by selecting the last points in the database.
    """

    def make_reduction(self, all_indices, **kwargs):
        "Use the last data points."
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Get the number of missing points
        npoints = int(self.npoints - len(indices))
        # Get the last points in the database
        if npoints > 0:
            indices = append(indices, not_indices[-npoints:])
        return array(indices, dtype=int)


class DatabaseRestart(DatabaseReduction):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduced data set is selected from restarts after npoints are used.
    The initial indices and the last data point is used at each restart.
    """

    def make_reduction(self, all_indices, **kwargs):
        "Make restart of used data set."
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the data set size
        data_len = len(all_indices)
        # Check how many last points are used
        lasts = self.include_last
        if lasts == 0:
            lasts = 1
        # Get the minimum number of points in the database
        n_initial = len(indices)
        if lasts > 1:
            n_initial += lasts - 1
        # Get the number of data point after the first restart
        n_use = data_len - self.npoints - 1
        # Get the number of points that are not initial or last indices
        nfree = self.npoints - n_initial
        # Get the excess of data points after each restart
        n_extra = int(n_use % nfree)
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the indices
        lasts_i = -(n_extra + lasts)
        indices = append(indices, not_indices[lasts_i:])
        return array(indices, dtype=int)


class DatabasePointsInterest(DatabaseLast):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduced data set is selected from the distances
    to the points of interest.
    The distance metric is the shortest distance
    to any of the points of interest.
    """

    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        round_targets=None,
        seed=None,
        dtype=float,
        npoints=25,
        initial_indices=[0],
        include_last=1,
        feature_distance=True,
        point_interest=[],
        **kwargs,
    ):
        """
        Initialize the database.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
            feature_distance: bool
                Whether to calculate the distance in feature space (True)
                or Cartesian coordinate space (False).
            point_interest: list
                A list of the points of interest as ASE Atoms instances.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
            feature_distance=feature_distance,
            point_interest=point_interest,
            **kwargs,
        )

    def get_feature_interest(self, **kwargs):
        """
        Get the fingerprints of the atoms of interest.

        Returns:
            array: A matrix array with the features or fingerprints of
                the points of interest.
        """
        if self.use_fingerprint:
            return array(
                [feature.get_vector() for feature in self.fp_interest],
                dtype=self.dtype,
            )
        return array(self.fp_interest, dtype=self.dtype)

    def get_positions(self, atoms_list, **kwargs):
        """
         Get the Cartesian coordinates of the atoms.

        Parameters:
             atoms_list: list or ASE Atoms
                 A list of ASE Atoms.

         Returns:
             list: A list of the positions of the atoms for each system.
        """
        return array(
            [atoms.get_positions().reshape(-1) for atoms in atoms_list],
            dtype=self.dtype,
        )

    def get_positions_interest(self, **kwargs):
        """
        Get the Cartesian coordinates of the atoms of interest.

        Returns:
            list: A list of the positions of the atoms of interest
                for each system.
        """
        return self.get_positions(self.point_interest)

    def get_all_positions(self, **kwargs):
        """
        Get the Cartesian coordinates of all the atoms in the database.

        Returns:
            list: A list of the positions of all the atoms in the database
                for each system.
        """
        return self.get_positions(self.get_all_data_atoms())

    def get_distances(self, not_indices, **kwargs):
        """
        Calculate the distances to the points of interest.

        Parameters:
            not_indices: list
                A list of indices that not used yet.

        Returns:
            array: The distances to the points of interest.
        """
        # Get either features or coordinates
        if self.feature_distance:
            # Get all the features
            features = self.get_all_feature_vectors()
            features_interest = self.get_feature_interest()
        else:
            # Get all the coordinates
            features = self.get_all_positions()
            features_interest = self.get_positions_interest()
        # Get the dimension
        fdim = len(features[0])
        # Calculate the minimum distances to the points of interest
        dist = cdist(
            features_interest, features[not_indices].reshape(-1, fdim)
        )
        return dist

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
        round_targets=None,
        seed=None,
        dtype=None,
        npoints=None,
        initial_indices=None,
        include_last=None,
        feature_distance=None,
        point_interest=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            fingerprint: Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint: bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            round_targets: int (optional)
                The number of decimals to round the targets to.
                If None, the targets are not rounded.
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            npoints: int
                Number of points that are used from the database.
            initial_indices: list
                The indices of the data points that must be included
                in the used data base.
            include_last: int
                Number of last data point to include in the used data base.
            feature_distance: bool
                Whether to calculate the distance in feature space (True)
                or Cartesian coordinate space (False).
            point_interest: list
                A list of the points of interest as ASE Atoms instances.

        Returns:
            self: The updated object itself.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=npoints,
            initial_indices=initial_indices,
            include_last=include_last,
        )
        # Set the feature distance
        if feature_distance is not None:
            self.feature_distance = feature_distance
        # Set the points of interest
        if point_interest is not None:
            # Ensure point_interest is a list of ASE Atoms instances
            if not isinstance(point_interest, list):
                point_interest = [point_interest]
            self.point_interest = [atoms.copy() for atoms in point_interest]
            self.fp_interest = [
                self.make_atoms_feature(atoms) for atoms in self.point_interest
            ]
        return self

    def make_reduction(self, all_indices, **kwargs):
        """
        Reduce the training set with the points closest to
        the points of interests.
        """
        # Check if there are points of interest else use the Parent class
        if len(self.point_interest) == 0:
            return super().make_reduction(all_indices, **kwargs)
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the last point
        indices = self.get_last_indices(indices, not_indices)
        # Get the indices for the system not already included
        not_indices = array(self.get_not_indices(indices, all_indices))
        # Get the number of missing points
        npoints = int(self.npoints - len(indices))
        # Calculate the distances to the points of interest
        dist = self.get_distances(not_indices)
        # Get the minimum distances to the points of interest
        dist = dist.min(axis=0)
        i_min = argsort(dist)[:npoints]
        # Get the indices
        indices = append(indices, [not_indices[i_min]])
        return array(indices, dtype=int)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            round_targets=self.round_targets,
            seed=self.seed,
            dtype=self.dtype,
            npoints=self.npoints,
            initial_indices=self.initial_indices,
            include_last=self.include_last,
            feature_distance=self.feature_distance,
            point_interest=self.point_interest,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indices=self.update_indices)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indices=self.indices.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabasePointsInterestEach(DatabasePointsInterest):
    """
    Database of ASE Atoms instances that are converted
    into stored fingerprints and targets.
    The used Database is a reduced set of the full Database.
    The reduced data set is selected from the distances
    to each point of interest.
    The distance metric is the shortest distance to the point of interest
    and it is performed iteratively.
    """

    def make_reduction(self, all_indices, **kwargs):
        """
        Reduce the training set with the points closest to
        the points of interests.
        """
        # Check if there are points of interest else use the Parent class
        if len(self.point_interest) == 0:
            return super().make_reduction(all_indices, **kwargs)
        # Get the fixed indices
        indices = self.get_initial_indices()
        # Get the indices for the system not already included
        not_indices = self.get_not_indices(indices, all_indices)
        # Include the last point
        indices = self.get_last_indices(indices, not_indices)
        # Get the indices for the system not already included
        not_indices = array(self.get_not_indices(indices, all_indices))
        # Calculate the distances to the points of interest
        dist = self.get_distances(not_indices)
        # Get the number of points of interest
        n_points_interest = len(dist)
        # Iterate over the points of interests
        p = 0
        while len(indices) < self.npoints:
            # Get the point with the minimum distance
            i_min = argmin(dist[p])
            # Get and append the index
            indices = append(indices, [not_indices[i_min]])
            # Remove the index
            not_indices = delete(not_indices, i_min)
            dist = delete(dist, i_min, axis=1)
            # Use the next point
            p += 1
            if p >= n_points_interest:
                p = 0
        return array(indices, dtype=int)
