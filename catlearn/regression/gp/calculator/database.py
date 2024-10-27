import numpy as np
from scipy.spatial.distance import cdist
from ase.constraints import FixAtoms
from ase.io.trajectory import TrajectoryWriter
from .copy_atoms import copy_atoms


class Database:
    def __init__(
        self,
        fingerprint=None,
        reduce_dimensions=True,
        use_derivatives=True,
        use_fingerprint=True,
        **kwargs,
    ):
        """
        Database of ASE atoms objects that are converted
        into fingerprints and targets.

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
        """
        # The negative forces have to be used since the derivatives are used
        self.use_negative_forces = True
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
            **kwargs,
        )

    def add(self, atoms, **kwargs):
        """
        Add an ASE Atoms object to the database.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            self: The updated object itself.
        """
        self.append(atoms, **kwargs)
        return self

    def add_set(self, atoms_list, **kwargs):
        """
        Add a set of ASE Atoms objects to the database.

        Parameters:
            atoms_list : list or ASE Atoms
                A list of or a single ASE Atoms
                with calculated energies and forces.

        Returns:
            self: The updated object itself.
        """
        for atoms in atoms_list:
            self.append(atoms, **kwargs)
        return self

    def get_constraints(self, atoms, **kwargs):
        """
        Get the indicies of the atoms that does not have fixed constraints.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            not_masked : list
                A list of indicies for the moving atoms.
        """
        not_masked = list(range(len(atoms)))
        if not self.reduce_dimensions:
            return not_masked
        constraints = atoms.constraints
        if len(constraints):
            masked = [
                c.get_indices() for c in constraints if isinstance(c, FixAtoms)
            ]
            if len(masked):
                masked = set(np.concatenate(masked))
                return list(set(not_masked).difference(masked))
        return not_masked

    def get_data_atoms(self, **kwargs):
        """
        Get the list of atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return [self.copy_atoms(atoms) for atoms in self.atoms_list]

    def get_features(self, **kwargs):
        """
        Get all the fingerprints of the atoms in the database.

        Returns:
            array: A matrix array with the saved features or fingerprints.
        """
        return np.array(self.features)

    def get_targets(self, **kwargs):
        """
        Get all the targets of the atoms in the database.

        Returns:
            array: A matrix array with the saved targets.
        """
        return np.array(self.targets)

    def save_data(self, trajectory="data.traj", mode="w", **kwargs):
        """
        Save the ASE Atoms data to a trajectory.

        Parameters:
            trajectory : str or TrajectoryWriter instance
                The name of the trajectory file where the data is saved.
                Or a TrajectoryWriter instance where the data is saved to.
            mode : str
                The mode of the trajectory file.

        Returns:
            self: The updated object itself.
        """
        if trajectory is None:
            return self
        if isinstance(trajectory, str):
            with TrajectoryWriter(trajectory, mode=mode) as traj:
                for atoms in self.atoms_list:
                    traj.write(atoms)
        elif isinstance(trajectory, TrajectoryWriter):
            for atoms in self.atoms_list:
                trajectory.write(atoms)
        return self

    def copy_atoms(self, atoms, **kwargs):
        """
        Copy the atoms object together with the calculated properties.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator that is copied.

        Returns:
            ASE Atoms:
                The copy of the Atoms object with saved data in the calculator.
        """
        return copy_atoms(atoms)

    def make_atoms_feature(self, atoms, **kwargs):
        """
        Make the feature or fingerprint of a single Atoms object.
        It can e.g. be used for predicting.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            fingerprint object: The fingerprint object of the Atoms object.
            or
            array: The feature or fingerprint array of the Atoms object.
        """
        if self.use_fingerprint:
            return self.fingerprint(atoms)
        return self.fingerprint(atoms).get_vector()

    def make_target(
        self,
        atoms,
        use_derivatives=True,
        use_negative_forces=True,
        **kwargs,
    ):
        """
        Calculate the target as the energy and forces if selected.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_negative_forces : bool
                Whether derivatives (True) or forces (False) are used.

        Returns:
            (1) array: The energy of the atoms object
                if use_derivatives=False.
            or
            (1,1+3*Nat) array: The energy and derivatives
                if use_derivatives=True.
        """
        e = atoms.get_potential_energy()
        if use_derivatives:
            not_masked = self.get_constraints(atoms)
            f = atoms.get_forces(apply_constraint=False)
            f = f[not_masked].reshape(-1)
            if use_negative_forces:
                return np.concatenate([[e], -f]).reshape(-1)
            return np.concatenate([[e], f]).reshape(-1)
        return np.array([e])

    def reset_database(self, **kwargs):
        """
        Reset the database by emptying the lists.

        Returns:
            self: The updated object itself.
        """
        self.atoms_list = []
        self.features = []
        self.targets = []
        return self

    def is_in_database(self, atoms, dtol=1e-8, **kwargs):
        """
        Check if the ASE Atoms is in the database.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.
            dtol : float
                The tolerance value to determine identical Atoms.

        Returns:
            bool: Whether the ASE Atoms object is within the database.
        """
        # Make the atoms object into a fingerprint
        fp_atoms = self.make_atoms_feature(atoms)
        # Get the fingerprints of the atoms in the database
        fp_database = self.get_features()
        # Check if the database is empty
        if len(fp_database) == 0:
            return False
        # Transform the fingerprints into vectors
        if self.use_fingerprint:
            fp_atoms = fp_atoms.get_vector()
            fp_database = np.array([fp.get_vector() for fp in fp_database])
        # Get the minimum distance between atoms object and the database
        dis_min = np.min(cdist([fp_atoms], fp_database))
        # Check if the atoms object is in the database
        if dis_min < dtol:
            return True
        return False

    def append(self, atoms, **kwargs):
        "Append the atoms object, the fingerprint, and target(s) to lists."
        # Copy the Atoms object
        atoms = self.copy_atoms(atoms)
        # Append the Atoms object
        self.atoms_list.append(atoms)
        # Append the feature
        self.features.append(self.make_atoms_feature(atoms))
        # Append the target(s)
        target = self.make_target(
            atoms,
            use_derivatives=self.use_derivatives,
            use_negative_forces=self.use_negative_forces,
        )
        self.targets.append(target)
        return self

    def get_use_derivatives(self):
        "Get whether the derivatives of the targets are used."
        return self.use_derivatives

    def get_reduce_dimensions(self):
        """
        Get whether the reduction of the fingerprint space is used
        if constrains are used.
        """
        return self.reduce_dimensions

    def get_use_fingerprint(self):
        "Get whether a fingerprint is used as the features."
        return self.use_fingerprint

    def update_arguments(
        self,
        fingerprint=None,
        reduce_dimensions=None,
        use_derivatives=None,
        use_fingerprint=None,
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
        # Check that the database and the fingerprint have the same attributes
        self.check_attributes()
        # Reset the database if an argument has been changed
        if reset_database:
            self.reset_database()
        return self

    def check_attributes(self):
        "Check if all attributes agree between the class and subclasses."
        if self.reduce_dimensions != self.fingerprint.get_reduce_dimensions():
            raise Exception(
                "Database and Fingerprint do not agree "
                "whether to reduce dimensions!"
            )
        return True

    def __len__(self):
        "Get the number of atoms objects in the database."
        return len(self.atoms_list)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            fingerprint=self.fingerprint,
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            use_fingerprint=self.use_fingerprint,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict(
            atoms_list=self.atoms_list.copy(),
            features=self.features.copy(),
            targets=self.targets.copy(),
        )
        return arg_kwargs, constant_kwargs, object_kwargs

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)
