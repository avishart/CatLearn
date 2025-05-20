from numpy import ndarray
from .mlmodel import MLModel
from .mlcalc import MLCalculator


class HierarchicalMLModel(MLModel):
    """
    Machine Learning model used for the ASE Atoms instances and
    in the machine learning calculators.
    It is a hierarchy of ML models where the first model is used
    for the first npoints data points. A new model is made when the
    number of data points exceed the number of points.
    The old models are used as a baseline.
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
        verbose=False,
        npoints=25,
        initial_indicies=[0],
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
            verbose: bool
                Whether to print statements in the optimization.
            npoints: int
                Number of points that are used from the database in the models.
            initial_indicies: list
                The indicies of the data points that must be included in
                the used data base for every model.
            dtype: type
                The data type of the arrays.
        """
        super().__init__(
            model=model,
            database=database,
            baseline=baseline,
            optimize=optimize,
            hp=hp,
            pdis=pdis,
            include_noise=include_noise,
            verbose=verbose,
            npoints=npoints,
            initial_indicies=initial_indicies,
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
        data_len = self.get_training_set_size()
        if not isinstance(atoms_list, (list, ndarray)):
            atoms_list = [atoms_list]
        # Store the data
        if data_len + len(atoms_list) <= self.npoints:
            # Include data in the same model
            super().add_training(atoms_list)
        elif data_len == self.npoints and len(atoms_list) == 1:
            # Make the current ml model into the new baseline
            self.baseline = MLCalculator(
                mlmodel=self.copy(),
                calculate_forces=True,
                calculate_uncertainty=False,
            )
            # Make a new ml model with the mandatory points
            data_atoms = self.get_data_atoms()
            data_atoms = [data_atoms[i] for i in self.initial_indicies]
            self.reset_database()
            super().add_training(data_atoms)
            super().add_training(atoms_list)
        else:
            raise AttributeError(
                "New baseline model can not be made without training. "
                "Include one point at the time!"
            )
        return self

    def update_arguments(
        self,
        model=None,
        database=None,
        baseline=None,
        optimize=None,
        hp=None,
        pdis=None,
        include_noise=None,
        verbose=None,
        npoints=None,
        initial_indicies=None,
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
            verbose: bool
                Whether to print statements in the optimization.
            npoints: int
                Number of points that are used from the database in the models.
            initial_indicies: list
                The indicies of the data points that must be included in
                the used data base for every model.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            model=model,
            database=database,
            baseline=baseline,
            optimize=optimize,
            hp=hp,
            pdis=pdis,
            include_noise=include_noise,
            verbose=verbose,
            dtype=dtype,
        )
        # Set the number of points
        if npoints is not None:
            self.npoints = int(npoints)
        # Set the initial indicies
        if initial_indicies is not None:
            self.initial_indicies = initial_indicies.copy()
        return self

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
            verbose=self.verbose,
            npoints=self.npoints,
            initial_indicies=self.initial_indicies,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict(baseline_targets=self.baseline_targets.copy())
        return arg_kwargs, constant_kwargs, object_kwargs
