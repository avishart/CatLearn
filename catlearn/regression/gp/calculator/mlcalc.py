from numpy import round as round_
from ase.calculators.calculator import Calculator, all_changes
from .mlmodel import MLModel


class MLCalculator(Calculator):
    """
    The machine learning calculator object applicable as an ASE calculator for
    ASE Atoms instance.
    """

    # Define the properties available in this calculator
    implemented_properties = [
        "energy",
        "forces",
        "uncertainty",
        "force uncertainties",
        "uncertainty derivatives",
    ]
    nolabel = True

    def __init__(
        self,
        mlmodel=None,
        calc_forces=True,
        calc_unc=True,
        calc_force_unc=False,
        calc_unc_deriv=False,
        calc_kwargs={},
        round_pred=None,
        **kwargs,
    ):
        """
        Initialize the ML calculator.

        Parameters:
            mlmodel: MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model,
                and add_training.
            calc_forces: bool
                Whether to calculate the forces.
            calc_unc: bool
                Whether to calculate
                the uncertainty prediction of the energy.
            calc_force_unc: bool
                Whether to calculate
                the uncertainties of the force predictions.
            calc_unc_deriv: bool
                Whether to calculate
                the derivatives of the uncertainty of the energy.
            calc_kwargs: dict
                A dictionary with kwargs for
                the parent calculator class object.
            round_pred: int (optional)
                The number of decimals to round the predictions to.
                If None, the predictions are not rounded.
        """
        # Inherit from the Calculator object
        Calculator.__init__(self, **calc_kwargs)
        # Set default mlmodel
        if mlmodel is None:
            mlmodel = MLModel(
                model=None,
                database=None,
                baseline=None,
                optimize=True,
            )
        # Set all the arguments
        self.update_arguments(
            mlmodel=mlmodel,
            calc_forces=calc_forces,
            calc_unc=calc_unc,
            calc_force_unc=calc_force_unc,
            calc_unc_deriv=calc_unc_deriv,
            calc_kwargs=calc_kwargs,
            round_pred=round_pred,
            **kwargs,
        )

    def get_uncertainty(self, atoms=None, **kwargs):
        """
        Get the predicted uncertainty of the energy.

        Parameters:
            atoms: ASE Atoms (optional)
                The ASE Atoms instance which is used
                if the uncertainty is not stored.

        Returns:
            float: The predicted uncertainty of the energy.
        """
        return self.get_property("uncertainty", atoms=atoms)

    def get_force_uncertainty(self, atoms=None, **kwargs):
        """
        Get the predicted uncertainty of the forces.

        Parameters:
            atoms: ASE Atoms (optional)
                The ASE Atoms instance which is used
                if the force uncertainties are not stored.

        Returns:
            (Nat,3) array: The predicted uncertainty of the forces.
        """
        return self.get_property("force uncertainties", atoms=atoms)

    def get_uncertainty_derivatives(self, atoms=None, **kwargs):
        """
        Get the derivatives of the uncertainty of the energy.

        Parameters:
            atoms: ASE Atoms (optional)
                The ASE Atoms instance which is used
                if the derivatives of the uncertainty are not stored.

        Returns:
            (Nat,3) array: The predicted uncertainty of the forces.
        """
        return self.get_property("uncertainty derivatives", atoms=atoms)

    def set_atoms(self, atoms, **kwargs):
        """
        Save the ASE Atoms instance in the calculator.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms instance that are saved.

        Returns:
            self: The updated object itself.
        """
        if self.check_state(atoms):
            self.reset()
            self.atoms = atoms.copy()
        return self

    def add_training(self, atoms_list, **kwarg):
        """
        Add training data as ASE Atoms to the ML model.

        Parameters:
            atoms_list: list or ASE Atoms
                A list of or a single ASE Atoms
                with calculated energies and forces.

        Returns:
            self: The updated object itself.
        """
        " Add training ase Atoms data to the ML model. "
        self.mlmodel.add_training(atoms_list, **kwarg)
        return self

    def train_model(self, **kwarg):
        """
        Train the ML model and optimize its hyperparameters if it is chosen.

        Returns:
            self: The updated object itself.
        """
        self.mlmodel.train_model(**kwarg)
        return self

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
        self.mlmodel.save_data(
            trajectory=trajectory,
            mode=mode,
            write_last=write_last,
            **kwargs,
        )
        return self

    def get_data_atoms(self, **kwargs):
        """
        Get the list of atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return self.mlmodel.get_data_atoms()

    def get_training_set_size(self):
        """
        Get the number of atoms objects in the ML model.

        Returns:
            int: The number of atoms objects in the ML model.
        """
        return self.mlmodel.get_training_set_size()

    def is_in_database(self, atoms, **kwargs):
        """
        Check if the ASE Atoms is in the database.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms instance with a calculator.

        Returns:
            bool: Whether the ASE Atoms instance is within the database.
        """
        return self.mlmodel.is_in_database(atoms=atoms, **kwargs)

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
        return self.mlmodel.copy_atoms(atoms, **kwargs)

    def update_mlmodel_arguments(self, **kwargs):
        """
        Update the arguments in the ML model.

        Parameters:
            kwargs: dict
                A dictionary with the arguments to update.

        Returns:
            self: The updated object itself.
        """
        self.mlmodel.update_arguments(**kwargs)
        return self

    def update_database_arguments(self, point_interest=None, **kwargs):
        """
        Update the arguments in the database.

        Parameters:
            point_interest: list
                A list of the points of interest as ASE Atoms instances.

        Returns:
            self: The updated object itself.
        """
        self.mlmodel.update_database_arguments(
            point_interest=point_interest, **kwargs
        )
        return self

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """
        Calculate the prediction energy, forces, and uncertainties of
        the energies and forces for a given ASE Atoms structure.
        Predicted potential energy can be obtained
        by *atoms.get_potential_energy()*,
        predicted forces using *atoms.get_forces()*,
        uncertainty of the energy using *atoms.calc.get_uncertainty(atoms)*,
        uncertainties of the forces
        using *atoms.calc.get_force_uncertainty(atoms)*,
        and derivatives of uncertainty
        using *atoms.calc.get_uncertainty_derivatives(atoms)*.

        Returns:
            self.results: dict
                A dictionary with all the calculated properties.
        """
        # Atoms object
        Calculator.calculate(self, atoms, properties, system_changes)
        # Get the arguments for calculating the requested properties
        (
            get_forces,
            get_uncertainty,
            get_force_uncertainties,
            get_unc_derivatives,
        ) = self.get_property_arguments(properties)
        # Get predict energy, forces and uncertainties for the given geometry
        results = self.model_prediction(
            atoms,
            get_forces=get_forces,
            get_uncertainty=get_uncertainty,
            get_force_uncertainties=get_force_uncertainties,
            get_unc_derivatives=get_unc_derivatives,
        )
        # Store the properties that are implemented
        self.store_properties(results)
        return self.results

    def save_mlcalc(self, filename="mlcalc.pkl", **kwargs):
        """
        Save the ML calculator object to a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            self: The object itself.
        """
        import pickle

        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return self

    def load_mlcalc(self, filename="mlcalc.pkl", **kwargs):
        """
        Load the ML calculator object from a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            mlcalc: The loaded ML calculator object.
        """
        import pickle

        with open(filename, "rb") as file:
            mlcalc = pickle.load(file)
        return mlcalc

    def update_arguments(
        self,
        mlmodel=None,
        calc_forces=None,
        calc_unc=None,
        calc_force_unc=None,
        calc_unc_deriv=None,
        calc_kwargs=None,
        round_pred=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            mlmodel: MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model,
                and add_training.
            calc_forces: bool
                Whether to calculate the forces.
            calc_unc: bool
                Whether to calculate
                the uncertainty prediction of the energy.
            calc_force_unc: bool
                Whether to calculate
                the uncertainties of the force predictions.
            calc_unc_deriv: bool
                Whether to calculate
                the derivatives of the uncertainty of the energy.
            calc_kwargs: dict
                A dictionary with kwargs for
                the parent calculator class object.
            round_pred: int (optional)
                The number of decimals to round the predictions to.
                If None, the predictions are not rounded.

        Returns:
            self: The updated object itself.
        """
        reset = False
        if mlmodel is not None:
            self.mlmodel = mlmodel.copy()
            reset = True
        if calc_forces is not None:
            self.calc_forces = calc_forces
            reset = True
        if calc_unc is not None:
            self.calc_unc = calc_unc
            reset = True
        if calc_force_unc is not None:
            self.calc_force_unc = calc_force_unc
            reset = True
        if calc_unc_deriv is not None:
            self.calc_unc_deriv = calc_unc_deriv
            reset = True
        if calc_kwargs is not None:
            self.calc_kwargs = calc_kwargs.copy()
            reset = True
        if round_pred is not None or not hasattr(self, "round_pred"):
            self.round_pred = round_pred
            reset = True
        # Empty the results
        if reset:
            self.reset()
        return self

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
        # Set the random seed for the ML model
        self.mlmodel.set_seed(seed)
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
        self.mlmodel.set_dtype(dtype, **kwargs)
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
        self.mlmodel.set_use_fingerprint(use_fingerprint=use_fingerprint)
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
        self.mlmodel.set_use_derivatives(use_derivatives=use_derivatives)
        return self

    def get_property_arguments(self, properties=[], **kwargs):
        """
        Get the arguments that ensure calculations of the properties requested.
        """
        # Check if the forces must be predicted
        if self.calc_forces or "forces" in properties:
            get_forces = True
        else:
            get_forces = False
        # Check if the uncertainty must be predicted
        if self.calc_unc or "uncertainty" in properties:
            get_uncertainty = True
        else:
            get_uncertainty = False
        # Check if the force uncertainties must be predicted
        if self.calc_force_unc or "force uncertainties" in properties:
            get_force_uncertainties = True
        else:
            get_force_uncertainties = False
        # Check if the derivatives of the uncertainty must be predicted
        if self.calc_unc_deriv or "uncertainty derivatives" in properties:
            get_unc_derivatives = True
        else:
            get_unc_derivatives = False
        return (
            get_forces,
            get_uncertainty,
            get_force_uncertainties,
            get_unc_derivatives,
        )

    def model_prediction(
        self,
        atoms,
        get_forces=False,
        get_uncertainty=False,
        get_force_uncertainties=False,
        get_unc_derivatives=False,
        **kwargs,
    ):
        """
        Predict energy, forces and uncertainties for the given geometry
        with the ML model.
        """
        results = self.mlmodel.calculate(
            atoms,
            get_forces=get_forces,
            get_uncertainty=get_uncertainty,
            get_force_uncertainties=get_force_uncertainties,
            get_unc_derivatives=get_unc_derivatives,
            **kwargs,
        )
        return results

    def store_properties(self, results, **kwargs):
        "Store the properties that are implemented."
        for key, value in results.items():
            if key in self.implemented_properties:
                # Round the predictions if needed
                if self.round_pred is not None:
                    if isinstance(value, float):
                        value = round(value, self.round_pred)
                    else:
                        value = round_(value, self.round_pred)
                # Save the properties in the results
                self.results[key] = value
        return self.results

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            mlmodel=self.mlmodel,
            calc_forces=self.calc_forces,
            calc_unc=self.calc_unc,
            calc_force_unc=self.calc_force_unc,
            calc_unc_deriv=self.calc_unc_deriv,
            calc_kwargs=self.calc_kwargs,
            round_pred=self.round_pred,
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

    def __deepcopy__(self, memo):
        "Do a deepcopy by using the copy function to make a new object."
        return self.copy()
