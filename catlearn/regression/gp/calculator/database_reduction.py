import numpy as np
from scipy.spatial.distance import cdist
from .database import Database


class DatabaseReduction(Database):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from a method.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
        """
        # The negative forces have to be used since the derivatives are used
        self.use_negative_forces = True
        # Set initial indicies
        self.indicies = []
        # Use default fingerprint if it is not given
        if fingerprint is None:
            from ..fingerprint.cartesian import Cartesian

            fingerprint = Cartesian(
                reduce_dimensions=reduce_dimensions,
                use_derivatives=use_derivatives,
            )
        # Set the arguments
        self.update_arguments(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
            include_last=include_last,
            **kwargs,
        )

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
        npoints=None,
        initial_indicies=None,
        include_last=None,
        **kwargs,
    ):
        """
        Update the class with its arguments. The existing arguments are used
        if they are not given.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.

        Returns:
            self: The updated object itself.
        """
        # Control if the database has to be reset
        reset_database = False
        if fingerprint is not None:
            self.fingerprint = fingerprint.copy()
            reset_database = True
        if reduce_dimensions is not None:
            self.reduce_dimensions = reduce_dimensions
            reset_database = True
        if use_derivatives is not None:
            self.use_derivatives = use_derivatives
            reset_database = True
        if use_fingerprint is not None:
            self.use_fingerprint = use_fingerprint
            reset_database = True
        if npoints is not None:
            self.npoints = int(npoints)
        if initial_indicies is not None:
            self.initial_indicies = np.array(initial_indicies, dtype=int)
        if include_last is not None:
            self.include_last = int(abs(include_last))
        # Check that too many last points are not included
        n_extra = self.npoints - len(self.initial_indicies)
        if self.include_last > n_extra:
            self.include_last = n_extra if n_extra >= 0 else 0
        # Check that the database and the fingerprint have the same attributes
        self.check_attributes()
        # Reset the database if an argument has been changed
        if reset_database:
            self.reset_database()
        # Store that the data base has changed
        self.update_indicies = True
        return self

    def get_all_atoms(self, **kwargs):
        """
        Get the list of all atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return self.atoms_list.copy()

    def get_atoms(self, **kwargs):
        """
        Get the list of atoms in the reduced database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        indicies = self.get_reduction_indicies()
        return [
            atoms
            for i, atoms in enumerate(self.get_all_atoms(**kwargs))
            if i in indicies
        ]

    def get_features(self, **kwargs):
        """
        Get the fingerprints of the atoms in the reduced database.

        Returns:
            array: A matrix array with the saved features or fingerprints.
        """
        indicies = self.get_reduction_indicies()
        return np.array(self.features)[indicies]

    def get_all_feature_vectors(self, **kwargs):
        "Get all the features in numpy array form."
        if self.use_fingerprint:
            features = [feature.get_vector() for feature in self.features]
            return np.array(features)
        return np.array(self.features)

    def get_targets(self, **kwargs):
        """
        Get the targets of the atoms in the reduced database.

        Returns:
            array: A matrix array with the saved targets.
        """
        indicies = self.get_reduction_indicies()
        return np.array(self.targets)[indicies]

    def get_all_targets(self, **kwargs):
        """
        Get all the targets of the atoms in the database.

        Returns:
            array: A matrix array with the saved targets.
        """
        return np.array(self.targets)

    def get_initial_indicies(self, **kwargs):
        """
        Get the initial indicies of the used atoms in the database.

        Returns:
            array: The initial indicies of the atoms used.
        """
        return self.initial_indicies.copy()

    def get_last_indicies(self, indicies, not_indicies, **kwargs):
        """
        Include the last indicies that are not in the used indicies list.

        Parameters:
            indicies : list
                A list of used indicies.
            not_indicies : list
                A list of indicies that not used yet.

        Returns:
            list: A list of the used indicies including the last indicies.
        """
        if self.include_last != 0:
            indicies = np.append(
                indicies,
                [not_indicies[-self.include_last :]],
            )
        return indicies

    def get_not_indicies(self, indicies, all_indicies, **kwargs):
        """
        Get a list of the indicies that are not in the used indicies list.

        Parameters:
            indicies : list
                A list of indicies.
            all_indicies : list
                A list of all indicies.

        Returns:
            list: A list of indicies that not used.
        """
        return list(set(all_indicies).difference(indicies))

    def append(self, atoms, **kwargs):
        "Append the atoms object, the fingerprint, and target(s) to lists."
        # Store that the data base has changed
        self.update_indicies = True
        # Append to the data base
        super().append(atoms, **kwargs)
        return self

    def get_reduction_indicies(self, **kwargs):
        "Get the indicies of the reduced data used."
        # If the indicies is already calculated then give them
        if not self.update_indicies:
            return self.indicies
        # Set up all the indicies
        self.update_indicies = False
        data_len = self.__len__()
        all_indicies = np.arange(data_len)
        # No reduction is needed if the database is not large
        if data_len <= self.npoints:
            self.indicies = all_indicies.copy()
            return self.indicies
        # Reduce the data base
        self.indicies = self.make_reduction(all_indicies)
        return self.indicies

    def make_reduction(self, all_indicies, **kwargs):
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
            npoints=self.npoints,
            initial_indicies=self.initial_indicies,
            include_last=self.include_last,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indicies=self.update_indicies)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indicies=self.indicies.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabaseDistance(DatabaseReduction):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from the distances.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
            include_last=include_last,
            **kwargs,
        )

    def make_reduction(self, all_indicies, **kwargs):
        "Reduce the training set with the points farthest from each other."
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the last point
        indicies = self.get_last_indicies(indicies, not_indicies)
        # Get a random index if no fixed index exist
        if len(indicies) == 0:
            indicies = np.array([np.random.choice(all_indicies)])
        # Get all the features
        features = self.get_all_feature_vectors()
        fdim = len(features[0])
        for i in range(len(indicies), self.npoints):
            # Get the indicies for the system not already included
            not_indicies = self.get_not_indicies(indicies, all_indicies)
            # Calculate the distances to the points already used
            dist = cdist(
                features[indicies].reshape(-1, fdim),
                features[not_indicies].reshape(-1, fdim),
            )
            # Choose the point furthest from the points already used
            i_max = np.argmax(np.nanmin(dist, axis=0))
            indicies = np.append(indicies, [not_indicies[i_max]])
        return np.array(indicies, dtype=int)


class DatabaseRandom(DatabaseReduction):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from random.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
            include_last=include_last,
            **kwargs,
        )

    def make_reduction(self, all_indicies, **kwargs):
        "Random select the training points."
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the last point
        indicies = self.get_last_indicies(indicies, not_indicies)
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Get the number of missing points
        npoints = int(self.npoints - len(indicies))
        # Randomly get the indicies
        indicies = np.append(
            indicies,
            np.random.permutation(not_indicies)[:npoints],
        )
        return np.array(indicies, dtype=int)


class DatabaseHybrid(DatabaseReduction):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        random_fraction=3,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from a mix of
        the distances and random.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            random_fraction : int
                How often the data point is sampled randomly.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
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
        npoints=None,
        initial_indicies=None,
        include_last=None,
        random_fraction=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            random_fraction : int
                How often the data point is sampled randomly.

        Returns:
            self: The updated object itself.
        """
        # Control if the database has to be reset
        reset_database = False
        if fingerprint is not None:
            self.fingerprint = fingerprint.copy()
            reset_database = True
        if reduce_dimensions is not None:
            self.reduce_dimensions = reduce_dimensions
            reset_database = True
        if use_derivatives is not None:
            self.use_derivatives = use_derivatives
            reset_database = True
        if use_fingerprint is not None:
            self.use_fingerprint = use_fingerprint
            reset_database = True
        if npoints is not None:
            self.npoints = int(npoints)
        if initial_indicies is not None:
            self.initial_indicies = np.array(initial_indicies, dtype=int)
        if include_last is not None:
            self.include_last = int(abs(include_last))
        if random_fraction is not None:
            self.random_fraction = int(abs(random_fraction))
            if self.random_fraction == 0:
                self.random_fraction = 1
        # Check that too many last points are not included
        n_extra = self.npoints - len(self.initial_indicies)
        if self.include_last > n_extra:
            self.include_last = n_extra if n_extra >= 0 else 0
        # Check that the database and the fingerprint have the same attributes
        self.check_attributes()
        # Reset the database if an argument has been changed
        if reset_database:
            self.reset_database()
        # Store that the data base has changed
        self.update_indicies = True
        return self

    def make_reduction(self, all_indicies, **kwargs):
        """
        Use a combination of random sampling and
        farthest distance to reduce training set.
        """
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the last point
        indicies = self.get_last_indicies(indicies, not_indicies)
        # Get a random index if no fixed index exist
        if len(indicies) == 0:
            indicies = [np.random.choice(all_indicies)]
        # Get all the features
        features = self.get_all_feature_vectors()
        fdim = len(features[0])
        for i in range(len(indicies), self.npoints):
            # Get the indicies for the system not already included
            not_indicies = self.get_not_indicies(indicies, all_indicies)
            if i % self.random_fraction == 0:
                # Get a random index
                indicies = np.append(
                    indicies,
                    [np.random.choice(not_indicies)],
                )
            else:
                # Calculate the distances to the points already used
                dist = cdist(
                    features[indicies].reshape(-1, fdim),
                    features[not_indicies].reshape(-1, fdim),
                )
                # Choose the point furthest from the points already used
                i_max = np.argmax(np.nanmin(dist, axis=0))
                indicies = np.append(indicies, [not_indicies[i_max]])
        return np.array(indicies, dtype=int)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            npoints=self.npoints,
            initial_indicies=self.initial_indicies,
            include_last=self.include_last,
            random_fraction=self.random_fraction,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indicies=self.update_indicies)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indicies=self.indicies.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabaseMin(DatabaseReduction):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        force_targets=False,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from the smallest targets.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            force_targets : bool
                Whether to include the derivatives/forces in targets
                when the smallest targets are found.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
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
        npoints=None,
        initial_indicies=None,
        include_last=None,
        force_targets=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            force_targets : bool
                Whether to include the derivatives/forces in targets
                when the smallest targets are found.

        Returns:
            self: The updated object itself.
        """
        # Control if the database has to be reset
        reset_database = False
        if fingerprint is not None:
            self.fingerprint = fingerprint.copy()
            reset_database = True
        if reduce_dimensions is not None:
            self.reduce_dimensions = reduce_dimensions
            reset_database = True
        if use_derivatives is not None:
            self.use_derivatives = use_derivatives
            reset_database = True
        if use_fingerprint is not None:
            self.use_fingerprint = use_fingerprint
            reset_database = True
        if npoints is not None:
            self.npoints = int(npoints)
        if initial_indicies is not None:
            self.initial_indicies = np.array(initial_indicies, dtype=int)
        if include_last is not None:
            self.include_last = int(abs(include_last))
        if force_targets is not None:
            self.force_targets = force_targets
        # Check that too many last points are not included
        n_extra = self.npoints - len(self.initial_indicies)
        if self.include_last > n_extra:
            self.include_last = n_extra if n_extra >= 0 else 0
        # Check that the database and the fingerprint have the same attributes
        self.check_attributes()
        # Reset the database if an argument has been changed
        if reset_database:
            self.reset_database()
        # Store that the data base has changed
        self.update_indicies = True
        return self

    def make_reduction(self, all_indicies, **kwargs):
        "Use the targets with smallest norms in the training set."
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the last point
        indicies = self.get_last_indicies(indicies, not_indicies)
        # Get the indicies for the system not already included
        not_indicies = np.array(self.get_not_indicies(indicies, all_indicies))
        # Get the targets
        targets = self.get_all_targets()[not_indicies]
        # Get sorting of the targets
        if self.force_targets:
            # Get the points with the lowest norm of the targets
            i_sort = np.argsort(np.linalg.norm(targets, axis=1))
        else:
            # Get the points with the lowest energies
            i_sort = np.argsort(targets[:, 0])
        # Get the number of missing points
        npoints = int(self.npoints - len(indicies))
        # Get the indicies for the system not already included
        i_sort = i_sort[:npoints]
        indicies = np.append(indicies, not_indicies[i_sort])
        return np.array(indicies, dtype=int)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            npoints=self.npoints,
            initial_indicies=self.initial_indicies,
            include_last=self.include_last,
            force_targets=self.force_targets,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indicies=self.update_indicies)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indicies=self.indicies.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabaseLast(DatabaseReduction):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from the last data points.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
            include_last=include_last,
            **kwargs,
        )

    def make_reduction(self, all_indicies, **kwargs):
        "Use the last data points."
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Get the number of missing points
        npoints = int(self.npoints - len(indicies))
        # Get the last points in the database
        if npoints > 0:
            indicies = np.append(indicies, not_indicies[-npoints:])
        return np.array(indicies, dtype=int)


class DatabaseRestart(DatabaseReduction):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from restarts after npoints are used.
        The initial indicies and the last data point is used at each restart.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
            include_last=include_last,
            **kwargs,
        )

    def make_reduction(self, all_indicies, **kwargs):
        "Make restart of used data set."
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the data set size
        data_len = len(all_indicies)
        # Check how many last points are used
        lasts = self.include_last
        if lasts == 0:
            lasts = 1
        # Get the minimum number of points in the database
        n_initial = len(indicies)
        if lasts > 1:
            n_initial += lasts - 1
        # Get the number of data point after the first restart
        n_use = data_len - self.npoints - 1
        # Get the number of points that are not initial or last indicies
        nfree = self.npoints - n_initial
        # Get the excess of data points after each restart
        n_extra = int(n_use % nfree)
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the indicies
        indicies = np.append(indicies, not_indicies[-(n_extra + lasts) :])
        return np.array(indicies, dtype=int)


class DatabasePointsInterest(DatabaseLast):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        feature_distance=True,
        point_interest=[],
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from the distances
        to the points of interest.
        The distance metric is the shortest distance
        to any of the points of interest.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            feature_distance : bool
                Whether to calculate the distance in feature space (True)
                or Cartesian coordinate space (False).
            point_interest : list
                A list of the points of interest as ASE Atoms instances.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
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
            return np.array(
                [feature.get_vector() for feature in self.fp_interest]
            )
        return np.array(self.fp_interest)

    def get_positions(self, atoms_list, **kwargs):
        """
         Get the Cartesian coordinates of the atoms.

        Parameters:
             atoms_list : list or ASE Atoms
                 A list of ASE Atoms.

         Returns:
             list: A list of the positions of the atoms for each system.
        """
        return np.array(
            [atoms.get_positions().reshape(-1) for atoms in atoms_list]
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
        return self.get_positions(self.get_all_atoms())

    def get_distances(self, not_indicies, **kwargs):
        """
        Calculate the distances to the points of interest.

        Parameters:
            not_indicies : list
                A list of indicies that not used yet.

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
            features_interest, features[not_indicies].reshape(-1, fdim)
        )
        return dist

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
        npoints=None,
        initial_indicies=None,
        include_last=None,
        feature_distance=None,
        point_interest=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            feature_distance : bool
                Whether to calculate the distance in feature space (True)
                or Cartesian coordinate space (False).
            point_interest : list
                A list of the points of interest as ASE Atoms instances.

        Returns:
            self: The updated object itself.
        """
        # Control if the database has to be reset
        reset_database = False
        if fingerprint is not None:
            self.fingerprint = fingerprint.copy()
            reset_database = True
        if reduce_dimensions is not None:
            self.reduce_dimensions = reduce_dimensions
            reset_database = True
        if use_derivatives is not None:
            self.use_derivatives = use_derivatives
            reset_database = True
        if use_fingerprint is not None:
            self.use_fingerprint = use_fingerprint
            reset_database = True
        if npoints is not None:
            self.npoints = int(npoints)
        if initial_indicies is not None:
            self.initial_indicies = np.array(initial_indicies, dtype=int)
        if include_last is not None:
            self.include_last = int(abs(include_last))
        if feature_distance is not None:
            self.feature_distance = feature_distance
        if point_interest is not None:
            self.point_interest = [atoms.copy() for atoms in point_interest]
            self.fp_interest = [
                self.make_atoms_feature(atoms) for atoms in self.point_interest
            ]
        # Check that too many last points are not included
        n_extra = self.npoints - len(self.initial_indicies)
        if self.include_last > n_extra:
            self.include_last = n_extra if n_extra >= 0 else 0
        # Check that the database and the fingerprint have the same attributes
        self.check_attributes()
        # Reset the database if an argument has been changed
        if reset_database:
            self.reset_database()
        # Store that the data base has changed
        self.update_indicies = True
        return self

    def make_reduction(self, all_indicies, **kwargs):
        """
        Reduce the training set with the points closest to
        the points of interests.
        """
        # Check if there are points of interest else use the Parent class
        if len(self.point_interest) == 0:
            return super().make_reduction(all_indicies, **kwargs)
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the last point
        indicies = self.get_last_indicies(indicies, not_indicies)
        # Get the indicies for the system not already included
        not_indicies = np.array(self.get_not_indicies(indicies, all_indicies))
        # Get the number of missing points
        npoints = int(self.npoints - len(indicies))
        # Calculate the distances to the points of interest
        dist = self.get_distances(not_indicies)
        # Get the minimum distances to the points of interest
        dist = np.min(dist, axis=0)
        i_min = np.argsort(dist)[:npoints]
        # Get the indicies
        indicies = np.append(indicies, [not_indicies[i_min]])
        return np.array(indicies, dtype=int)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
            npoints=self.npoints,
            initial_indicies=self.initial_indicies,
            include_last=self.include_last,
            feature_distance=self.feature_distance,
            point_interest=self.point_interest,
        )
        # Get the constants made within the class
        constant_kwargs = dict(update_indicies=self.update_indicies)
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
            indicies=self.indicies.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs


class DatabasePointsInterestEach(DatabasePointsInterest):
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        npoints=25,
        initial_indicies=[0],
        include_last=1,
        feature_distance=True,
        point_interest=[],
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.
        The used Database is a reduced set of the full Database.
        The reduced data set is selected from the distances
        to each point of interest.
        The distance metric is the shortest distance to the point of interest
        and it is performed iteratively.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class
                that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True)
                or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included
                in the used data base.
            include_last : int
                Number of last data point to include in the used data base.
            feature_distance : bool
                Whether to calculate the distance in feature space (True)
                or Cartesian coordinate space (False).
            point_interest : list
                A list of the points of interest as ASE Atoms instances.
        """
        super().__init__(
            fingerprint=fingerprint,
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            npoints=npoints,
            initial_indicies=initial_indicies,
            include_last=include_last,
            feature_distance=feature_distance,
            point_interest=point_interest,
            **kwargs,
        )

    def make_reduction(self, all_indicies, **kwargs):
        """
        Reduce the training set with the points closest to
        the points of interests.
        """
        # Check if there are points of interest else use the Parent class
        if len(self.point_interest) == 0:
            return super().make_reduction(all_indicies, **kwargs)
        # Get the fixed indicies
        indicies = self.get_initial_indicies()
        # Get the indicies for the system not already included
        not_indicies = self.get_not_indicies(indicies, all_indicies)
        # Include the last point
        indicies = self.get_last_indicies(indicies, not_indicies)
        # Get the indicies for the system not already included
        not_indicies = np.array(self.get_not_indicies(indicies, all_indicies))
        # Calculate the distances to the points of interest
        dist = self.get_distances(not_indicies)
        # Get the number of points of interest
        n_points_interest = len(dist)
        # Iterate over the points of interests
        p = 0
        while len(indicies) < self.npoints:
            # Get the point with the minimum distance
            i_min = np.argmin(dist[p])
            # Get and append the index
            indicies = np.append(indicies, [not_indicies[i_min]])
            # Remove the index
            not_indicies = np.delete(not_indicies, i_min)
            dist = np.delete(dist, i_min, axis=1)
            # Use the next point
            p += 1
            if p >= n_points_interest:
                p = 0
        return np.array(indicies, dtype=int)
