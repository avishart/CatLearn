import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms


class BaselineCalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(
        self,
        reduce_dimensions=True,
        **kwargs,
    ):
        """
        A baseline calculator for ASE atoms object.
        It uses a flat baseline with zero energy and forces.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions to only
                moving atoms if constrains are used.
        """
        super().__init__()
        self.update_arguments(reduce_dimensions=reduce_dimensions, **kwargs)

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
        if "forces" in properties:
            energy, forces = self.get_energy_forces(
                atoms,
                get_derivatives=True,
            )
            self.results["forces"] = forces
        else:
            energy = self.get_energy_forces(atoms, get_derivatives=False)
        self.results["energy"] = energy
        pass

    def update_arguments(self, reduce_dimensions=None, **kwargs):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms
                if constrains are used.
        Returns:
            self: The updated object itself.
        """
        if reduce_dimensions is not None:
            self.reduce_dimensions = reduce_dimensions
        return self

    def get_energy_forces(self, atoms, get_derivatives=True, **kwargs):
        "Get the energy and forces."
        if get_derivatives:
            return 0.0, np.zeros((len(atoms), 3))
        return 0.0

    def get_constraints(self, atoms, **kwargs):
        """
        Get the indicies of the atoms that does not have fixed constraints.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            not_masked : list
                A list of indicies for the moving atoms
                if constraints are used.
            masked : list
                A list of indicies for the fixed atoms
                if constraints are used.
        """
        not_masked = list(range(len(atoms)))
        if not self.reduce_dimensions:
            return not_masked, []
        constraints = atoms.constraints
        if len(constraints) > 0:
            masked = np.concatenate(
                [
                    c.get_indices()
                    for c in constraints
                    if isinstance(c, FixAtoms)
                ]
            )
            masked = set(masked)
            return list(set(not_masked).difference(masked)), list(masked)
        return not_masked, []

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(reduce_dimensions=self.reduce_dimensions)
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
