from numpy import asarray, ndarray, sqrt, zeros
from ase.parallel import parprint
import pickle
from .default_model import get_default_model, get_default_database


class MLModel:
    """
    Machine Learning model used for the ASE Atoms instances and
    in the machine learning calculators.
    """

    def __init__(
        self,
        model=None,
        database=None,
        baseline=None,
        optimize=True,
        hp=None,
        pdis=None,
        include_noise=False,
        to_save_mlmodel=False,
        save_mlmodel_kwargs={},
        verbose=False,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the ML model for Atoms.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
            database: Database object
                The Database object with ASE atoms.
            baseline: Baseline object
                The Baseline object calculator
                that calculates energy and forces.
            optimize: bool
                Whether to optimize the hyperparameters
                when the model is trained.
            hp: dict
                Use a set of hyperparameters to optimize from
                else the current set is used.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            include_noise: bool
                Whether to include noise in the uncertainty from the model.
            to_save_mlmodel: bool
                Whether to save the ML model to a file after training.
            save_mlmodel_kwargs: dict
                Arguments for saving the ML model, like the filename.
            verbose: bool
                Whether to print statements in the optimization.
            dtype: type
                The data type of the arrays.
        """
        # Make default model if it is not given
        if model is None:
            model = get_default_model(dtype=dtype)
        # Make default database if it is not given
        if database is None:
            database = get_default_database(dtype=dtype)
        # Set the arguments
        self.update_arguments(
            model=model,
            database=database,
            baseline=baseline,
            optimize=optimize,
            hp=hp,
            pdis=pdis,
            include_noise=include_noise,
            to_save_mlmodel=to_save_mlmodel,
            save_mlmodel_kwargs=save_mlmodel_kwargs,
            verbose=verbose,
            dtype=dtype,
            **kwargs,
        )

    def add_training(self, atoms_list, **kwargs):
        """
        Add training data in form of the ASE Atoms to the database.

        Parameters:
            atoms_list: list or ASE Atoms
                A list of or a single ASE Atoms with
                calculated energies and forces.

        Returns:
            self: The updated object itself.
        """
        if not isinstance(atoms_list, (list, ndarray)):
            atoms_list = [atoms_list]
        self.database.add_set(atoms_list)
        return self

    def train_model(self, **kwargs):
        """
        Train the ML model and optimize its hyperparameters if it is chosen.

        Returns:
            self: The updated object itself.
        """
        # Get data from the data base
        features, targets, atoms_list = self.get_data()
        # Correct targets with the baseline
        targets = self.get_baseline_corrected_targets(atoms_list, targets)
        # Train model
        if self.optimize:
            # Optimize the hyperparameters and train the ML model
            self.model_optimization(features, targets, **kwargs)
        else:
            # Train the ML model
            self.model_training(features, targets, **kwargs)
        # Save the ML model to a file if requested
        if self.to_save_mlmodel:
            self.save_mlmodel(**self.save_mlmodel_kwargs)
        return self

    def calculate(
        self,
        atoms,
        get_uncertainty=True,
        get_forces=True,
        get_force_uncertainties=False,
        get_unc_derivatives=False,
        **kwargs,
    ):
        """
        Calculate the energy and also the uncertainties and forces if selected.
        If get_variance=False, variance is returned as None.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms object that the properties (incl. energy)
                are calculated for.
            get_uncertainty: bool
                Whether to calculate the uncertainty.
                The uncertainty is None if get_uncertainty=False.
            get_forces: bool
                Whether to calculate the forces.
            get_force_uncertainties: bool
                Whether to calculate the uncertainties of the predicted forces.
            get_unc_derivatives: bool
                Whether to calculate the derivatives of
                the uncertainty of the predicted energy.

        Returns:
            energy: float
                The predicted energy of the ASE Atoms.
            forces: (Nat,3) array or None
                The predicted forces if get_forces=True.
            uncertainty: float or None
                The predicted uncertainty of the energy
                if get_uncertainty=True.
            uncertainty_forces: (Nat,3) array or None
                The predicted uncertainties of the forces
                if get_uncertainty=True and get_forces=True.
        """
        # Calculate energy, forces, and uncertainty
        energy, forces, unc, unc_forces, unc_deriv = self.model_prediction(
            atoms,
            get_uncertainty=get_uncertainty,
            get_forces=get_forces,
            get_force_uncertainties=get_force_uncertainties,
            get_unc_derivatives=get_unc_derivatives,
        )
        # Store the predictions
        results = self.store_results(
            atoms,
            energy=energy,
            forces=forces,
            unc=unc,
            unc_forces=unc_forces,
            unc_deriv=unc_deriv,
        )
        return results

    def save_data(
        self,
        trajectory="data.traj",
        mode="w",
        write_last=False,
        **kwargs,
    ):
        """
        Save the ASE Atoms data to a trajectory.

        Parameters:
            trajectory: str or TrajectoryWriter instance
                The name of the trajectory file where the data is saved.
                Or a TrajectoryWriter instance where the data is saved to.
            mode: str
                The mode of the trajectory file.
            write_last: bool
                Whether to only write the last atoms instance to the
                trajectory.
                If False, all atoms instances in the database are written
                to the trajectory.

        Returns:
            self: The updated object itself.
        """
        self.database.save_data(
            trajectory=trajectory,
            mode=mode,
            write_last=write_last,
            **kwargs,
        )
        return self

    def get_training_set_size(self, **kwargs):
        """
        Get the number of atoms objects in the database.

        Returns:
            int: The number of atoms objects in the database.
        """
        return len(self.database)

    def is_in_database(self, atoms, **kwargs):
        """
        Check if the ASE Atoms is in the database.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            bool: Whether the ASE Atoms object is within the database.
        """
        return self.database.is_in_database(atoms=atoms, **kwargs)

    def copy_atoms(self, atoms, **kwargs):
        """
        Copy the atoms object together with the calculated properties.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms object with a calculator that is copied.

        Returns:
            ASE Atoms: The copy of the Atoms object
            with saved data in the calculator.
        """
        return self.database.copy_atoms(atoms, **kwargs)

    def update_database_arguments(self, point_interest=None, **kwargs):
        """
        Update the arguments in the database.

        Parameters:
            point_interest: list
                A list of the points of interest as ASE Atoms instances.

        Returns:
            self: The updated object itself.
        """
        self.database.update_arguments(point_interest=point_interest, **kwargs)
        return self

    def save_mlmodel(self, filename="mlmodel.pkl", **kwargs):
        """
        Save the ML model instance to a file.

        Parameters:
            filename: str
                The name of the file where the instance is saved.

        Returns:
            self: The instance itself.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return self

    def load_mlmodel(self, filename="mlmodel.pkl", **kwargs):
        """
        Load the ML model instance from a file.

        Parameters:
            filename: str
                The name of the file where the instance is saved.

        Returns:
            mlcalc: The loaded ML model instance.
        """
        with open(filename, "rb") as file:
            mlmodel = pickle.load(file)
        return mlmodel

    def update_arguments(
        self,
        model=None,
        database=None,
        baseline=None,
        optimize=None,
        hp=None,
        pdis=None,
        include_noise=None,
        to_save_mlmodel=None,
        save_mlmodel_kwargs=None,
        verbose=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            model: Model
                The Machine Learning Model with kernel and
                prior that are optimized.
            database: Database object
                The Database object with ASE atoms.
            baseline: Baseline object
                The Baseline object calculator
                that calculates energy and forces.
            optimize: bool
                Whether to optimize the hyperparameters
                when the model is trained.
            hp: dict
                Use a set of hyperparameters to optimize from
                else the current set is used.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            include_noise: bool
                Whether to include noise in the uncertainty from the model.
            to_save_mlmodel: bool
                Whether to save the ML model to a file after training.
            save_mlmodel_kwargs: dict
                Arguments for saving the ML model, like the filename.
            verbose: bool
                Whether to print statements in the optimization.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        if model is not None:
            self.model = model.copy()
        if database is not None:
            self.database = database.copy()
        if baseline is not None:
            self.baseline = baseline.copy()
        elif not hasattr(self, "baseline"):
            self.baseline = None
        if optimize is not None:
            self.optimize = optimize
        if hp is not None:
            self.hp = hp.copy()
        elif not hasattr(self, "hp"):
            self.hp = None
        if pdis is not None:
            self.pdis = pdis.copy()
        elif not hasattr(self, "pdis"):
            self.pdis = None
        if include_noise is not None:
            self.include_noise = include_noise
        if to_save_mlmodel is not None:
            self.to_save_mlmodel = to_save_mlmodel
        if save_mlmodel_kwargs is not None:
            self.save_mlmodel_kwargs = save_mlmodel_kwargs
        if verbose is not None:
            self.verbose = verbose
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype=dtype)
        # Check if the baseline is used
        if self.baseline is None:
            self.use_baseline = False
        else:
            self.use_baseline = True
        # Check that the model and database have the same attributes
        self.check_attributes()
        return self

    def model_optimization(self, features, targets, **kwargs):
        "Optimize the ML model with the arguments set in optimize_kwargs."
        # Optimize the hyperparameters and train the ML model
        sol = self.model.optimize(
            features,
            targets,
            retrain=True,
            hp=self.hp,
            pdis=self.pdis,
            verbose=False,
            **kwargs,
        )
        # Print the solution if verbose is True
        if self.verbose:
            # Get the prefactor if it is available
            if hasattr(self.model, "get_prefactor"):
                sol["prefactor"] = float(
                    "{:.3e}".format(self.model.get_prefactor())
                )
            # Get the noise correction if it is available
            if hasattr(self.model, "get_correction"):
                sol["correction"] = float(
                    "{:.3e}".format(self.model.get_correction())
                )
            parprint(sol)
        return self.model

    def model_training(self, features, targets, **kwargs):
        "Train the model without optimizing the hyperparameters."
        self.model.train(features, targets, **kwargs)
        return self.model

    def model_prediction(
        self,
        atoms,
        get_uncertainty=True,
        get_forces=True,
        get_force_uncertainties=False,
        get_unc_derivatives=False,
        **kwargs,
    ):
        "Predict the targets and uncertainties."
        # Calculate fingerprint
        fp = self.make_atoms_feature(atoms)
        # Calculate energy, forces, and uncertainty
        y, var, var_deriv = self.model.predict(
            fp,
            get_derivatives=get_forces,
            get_variance=get_uncertainty,
            include_noise=self.include_noise,
            get_derivtives_var=get_force_uncertainties,
            get_var_derivatives=get_unc_derivatives,
        )
        # Correct the predicted targets with the baseline if it is used
        y = self.add_baseline_correction(
            y,
            atoms=atoms,
            use_derivatives=get_forces,
        )
        # Extract the energy
        energy = y.item(0)
        # Extract the forces if they are requested
        if get_forces:
            forces = -y[0][1:]
        else:
            forces = None
        # Get the uncertainties if they are requested
        if get_uncertainty:
            unc = sqrt(var.item(0))
            # Get the uncertainty of the forces if they are requested
            if get_force_uncertainties and get_forces:
                unc_forces = sqrt(var[0][1:])
            else:
                unc_forces = None
            # Get the derivatives of the predicted uncertainty
            if get_unc_derivatives:
                unc_deriv = (0.5 / unc) * var_deriv
            else:
                unc_deriv = None
        else:
            unc = None
            unc_forces = None
            unc_deriv = None
        return energy, forces, unc, unc_forces, unc_deriv

    def store_results(
        self,
        atoms,
        energy=None,
        forces=None,
        unc=None,
        unc_forces=None,
        unc_deriv=None,
        **kwargs,
    ):
        "Store the predicted results in a dictionary."
        results = {}
        # Save the energy
        if energy is not None:
            results["energy"] = energy
        # Save the uncertainty
        if unc is not None:
            results["uncertainty"] = unc
        # Get constraints
        if (
            forces is not None
            or unc_forces is not None
            or unc_deriv is not None
        ):
            natoms, not_masked = self.get_constraints(atoms)
        # Make the full matrix of forces and save it
        if forces is not None:
            results["forces"] = self.not_masked_reshape(
                forces,
                not_masked=not_masked,
                natoms=natoms,
            )
        # Make the full matrix of force uncertainties and save it
        if unc_forces is not None:
            results["force uncertainties"] = self.not_masked_reshape(
                unc_forces,
                not_masked=not_masked,
                natoms=natoms,
            )
        # Make the full matrix of derivatives of uncertainty and save it
        if unc_deriv is not None:
            results["uncertainty derivatives"] = self.not_masked_reshape(
                unc_deriv,
                not_masked=not_masked,
                natoms=natoms,
            )
        return results

    def add_baseline_correction(
        self,
        targets,
        atoms,
        use_derivatives=True,
        **kwargs,
    ):
        "Add the baseline correction to the targets if a baseline is used."
        if self.use_baseline:
            # Calculate the baseline for the ASE atoms instance
            y_base = self.calculate_baseline(
                [atoms],
                use_derivatives=use_derivatives,
                **kwargs,
            )
            # Add baseline correction to the targets
            targets += asarray(y_base, dtype=self.dtype)[0]
        return targets

    def get_baseline_corrected_targets(self, atoms_list, targets, **kwargs):
        """
        Get the baseline corrected targets if a baseline is used.
        The baseline correction is subtracted from training targets.
        """
        if self.use_baseline:
            # Calculate the baseline for each ASE atoms instance
            y_base = self.calculate_baseline(
                atoms_list,
                use_derivatives=self.database.use_derivatives,
                **kwargs,
            )
            # Subtract baseline correction to the targets
            targets -= asarray(y_base, dtype=self.dtype)
        return targets

    def calculate_baseline(self, atoms_list, use_derivatives=True, **kwargs):
        "Calculate the baseline for each ASE atoms object."
        y_base = []
        for atoms in atoms_list:
            atoms_base = atoms.copy()
            atoms_base.calc = self.baseline
            y_base.append(
                self.make_targets(
                    atoms_base,
                    use_derivatives=use_derivatives,
                    **kwargs,
                )
            )
        return y_base

    def not_masked_reshape(self, nm_array, not_masked, natoms, **kwargs):
        """
        Reshape an array so that it works for all atom coordinates and
        set constrained indices to 0.
        """
        full_array = zeros((natoms, 3), dtype=self.dtype)
        full_array[not_masked] = nm_array.reshape(-1, 3)
        return full_array

    def get_data(self, **kwargs):
        "Get data from the data base."
        features = self.database.get_features()
        targets = self.database.get_targets()
        atoms_list = self.get_data_atoms()
        return features, targets, atoms_list

    def get_data_atoms(self, **kwargs):
        """
        Get the list of atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return self.database.get_data_atoms()

    def reset_database(self, **kwargs):
        """
        Reset the database by emptying the lists.

        Returns:
            self: The updated object itself.
        """
        self.database.reset_database()
        return self

    def make_targets(self, atoms, use_derivatives=True, **kwargs):
        "Make the target in the data base."
        return self.database.make_target(
            atoms,
            use_derivatives=use_derivatives,
            use_negative_forces=True,
        )

    def get_constraints(self, atoms, **kwargs):
        """
        Get the number of atoms and the indices of
        the atoms without constraints.
        """
        natoms = len(atoms)
        not_masked = self.database.get_constraints(atoms, **kwargs)
        return natoms, not_masked

    def set_seed(self, seed=None, **kwargs):
        """
        Set the random seed.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        # Set the random seed for the database
        self.database.set_seed(seed)
        # Set the random seed for the model
        self.model.set_seed(seed)
        return self

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the data type
        self.dtype = dtype
        # Set the data type of the model and database
        self.model.set_dtype(dtype=dtype, **kwargs)
        self.database.set_dtype(dtype=dtype, **kwargs)
        # Set the data type of the baseline if it is used
        if self.baseline is not None:
            self.baseline.set_dtype(dtype=dtype, **kwargs)
        # Set the data type of the prior distributions if they are used
        if self.pdis is not None:
            for pdis in self.pdis.values():
                pdis.set_dtype(dtype=dtype, **kwargs)
        return self

    def set_use_fingerprint(self, use_fingerprint, **kwargs):
        """
        Set whether to use fingerprints in the model and database.

        Parameters:
            use_fingerprint: bool
                Whether to use fingerprints in the model and database.

        Returns:
            self: The updated object itself.
        """
        self.model.set_use_fingerprint(use_fingerprint=use_fingerprint)
        self.database.set_use_fingerprint(use_fingerprint=use_fingerprint)
        return self

    def set_use_derivatives(self, use_derivatives, **kwargs):
        """
        Set whether to use derivatives in the model and database.

        Parameters:
            use_derivatives: bool
                Whether to use derivatives in the model and database.

        Returns:
            self: The updated object itself.
        """
        self.model.set_use_derivatives(use_derivatives=use_derivatives)
        self.database.set_use_derivatives(use_derivatives=use_derivatives)
        # Set the data type of the baseline if it is used
        if self.baseline is not None:
            self.baseline.set_use_forces(use_derivatives)
        return self

    def make_atoms_feature(self, atoms, **kwargs):
        """
        Make the feature or fingerprint of a single Atoms object.
        It can e.g. be used for predicting.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            array of fingerprint object: The fingerprint object of the
            Atoms object.
            or
            array: The feature or fingerprint array of the Atoms object.
        """
        # Calculate fingerprint
        fp = self.database.make_atoms_feature(atoms, **kwargs)
        return asarray([fp])

    def check_attributes(self):
        "Check if all attributes agree between the class and subclasses."
        if (
            self.model.get_use_fingerprint()
            != self.database.get_use_fingerprint()
        ):
            raise ValueError(
                "Model and Database do not agree "
                "whether to use fingerprints!"
            )
        if (
            self.model.get_use_derivatives()
            != self.database.get_use_derivatives()
        ):
            raise ValueError(
                "Model and Database do not agree "
                "whether to use derivatives/forces!"
            )
        return True

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            model=self.model,
            database=self.database,
            baseline=self.baseline,
            optimize=self.optimize,
            hp=self.hp,
            pdis=self.pdis,
            include_noise=self.include_noise,
            to_save_mlmodel=self.to_save_mlmodel,
            save_mlmodel_kwargs=self.save_mlmodel_kwargs,
            verbose=self.verbose,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
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
