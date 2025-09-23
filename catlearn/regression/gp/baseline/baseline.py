from numpy import finfo, zeros
from ase.calculators.calculator import Calculator, all_changes


class BaselineCalculator(Calculator):
    """
    A baseline calculator for ASE Atoms instance.
    It uses a flat baseline with zero energy and forces.
    """

    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(
        self,
        reduce_dimensions=True,
        use_forces=True,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the baseline calculator.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions to only
                moving atoms if constrains are used.
            use_forces: bool
                Calculate and store the forces.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__()
        self.update_arguments(
            reduce_dimensions=reduce_dimensions,
            use_forces=use_forces,
            dtype=dtype,
            **kwargs,
        )

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """
        Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()*.
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Obtain energy and forces for the given structure:
        if "forces" in properties or self.use_forces:
            energy, forces = self.get_energy_forces(
                atoms,
                use_forces=True,
            )
            self.results["forces"] = forces
        else:
            energy, _ = self.get_energy_forces(atoms, use_forces=False)
        self.results["energy"] = energy
        pass

    def set_use_forces(self, use_forces, **kwargs):
        """
        Set whether to use forces or not.

        Parameters:
            use_forces: bool
                Whether to use forces or not.

        Returns:
            self: The updated object itself.
        """
        # Set the use_forces
        self.use_forces = use_forces
        return self

    def set_reduce_dimensions(self, reduce_dimensions, **kwargs):
        """
        Set whether to reduce the dimensions or not.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions or not.

        Returns:
            self: The updated object itself.
        """
        # Set the reduce_dimensions
        self.reduce_dimensions = reduce_dimensions
        return self

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        # Set the dtype
        self.dtype = dtype
        # Set a small number to avoid division by zero
        self.eps = finfo(self.dtype).eps
        return self

    def update_arguments(
        self,
        reduce_dimensions=None,
        use_forces=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms
                if constrains are used.
            use_forces: bool
                Calculate and store the forces.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        if reduce_dimensions is not None:
            self.set_reduce_dimensions(reduce_dimensions=reduce_dimensions)
        if use_forces is not None:
            self.set_use_forces(use_forces=use_forces)
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype=dtype)
        return self

    def get_energy_forces(self, atoms, use_forces=True, **kwargs):
        "Get the energy and forces."
        if use_forces:
            return 0.0, zeros((len(atoms), 3), dtype=self.dtype)
        return 0.0, None

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            reduce_dimensions=self.reduce_dimensions,
            use_forces=self.use_forces,
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
