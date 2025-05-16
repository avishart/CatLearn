from numpy import array, asarray, isscalar, ndarray
from ase.calculators.calculator import Calculator, PropertyNotImplementedError


def copy_atoms(atoms, results={}, **kwargs):
    """
    Copy the atoms instance together with the calculated properties.

    Parameters:
        atoms: ASE Atoms instance
            The ASE Atoms instance with a calculator that is copied.
        results: dict (optional)
            The properties to be saved in the calculator.
            If not given, the properties are taken from the calculator.

    Returns:
        atoms0: ASE Atoms instance
            The copy of the Atoms instance with saved data in the calculator.
    """
    # Check if results are given
    if not isinstance(results, dict) or len(results) == 0:
        # Save the properties calculated
        if atoms.calc is not None and atoms.calc.atoms is not None:
            if compare_atoms(atoms, atoms.calc.atoms):
                results = atoms.calc.results.copy()
    # Copy the ASE Atoms instance
    atoms0 = atoms.copy()
    # Store the properties in a calculator
    atoms0.calc = StoredDataCalculator(atoms, **results)
    return atoms0


def compare_atoms(
    atoms0,
    atoms1,
    tol=1e-8,
    properties_to_check=["atoms", "positions"],
    **kwargs,
):
    """
    Compare two atoms instances.

    Parameters:
        atoms0: ASE Atoms instance
            The first ASE Atoms instance.
        atoms1: ASE Atoms
            The second ASE Atoms instance.
        tol: float (optional)
            The tolerance for the comparison.
        properties_to_check: list (optional)
            The properties to be compared.

    Returns:
        bool: True if the atoms instances are equal otherwise False.
    """
    # Check if the number of atoms is equal
    if len(atoms0) != len(atoms1):
        return False
    # Check if the chemical symbols are equal
    if "atoms" in properties_to_check:
        if not (
            asarray(atoms0.get_chemical_symbols())
            == asarray(atoms1.get_chemical_symbols())
        ).all():
            return False
    # Check if the positions are equal
    if "positions" in properties_to_check:
        if abs(atoms0.get_positions() - atoms1.get_positions()).max() > tol:
            return False
    # Check if the cell is equal
    if "cell" in properties_to_check:
        if abs(atoms0.get_cell() - atoms1.get_cell()).max() > tol:
            return False
    # Check if the pbc is equal
    if "pbc" in properties_to_check:
        if not (asarray(atoms0.get_pbc()) == asarray(atoms1.get_pbc())).all():
            return False
    # Check if the initial charges are equal
    if "initial_charges" in properties_to_check:
        if (
            abs(
                atoms0.get_initial_charges() - atoms1.get_initial_charges()
            ).max()
            > tol
        ):
            return False
    # Check if the initial magnetic moments are equal
    if "initial_magnetic_moments" in properties_to_check:
        if (
            abs(
                atoms0.get_initial_magnetic_moments()
                - atoms1.get_initial_magnetic_moments()
            ).max()
            > tol
        ):
            return False
    # Check if the momenta are equal
    if "momenta" in properties_to_check:
        if abs(atoms0.get_momenta() - atoms1.get_momenta()).max() > tol:
            return False
    # Check if the velocities are equal
    if "velocities" in properties_to_check:
        if abs(atoms0.get_velocities() - atoms1.get_velocities()).max() > tol:
            return False
    return True


class StoredDataCalculator(Calculator):
    """
    A special calculator that store the data (results)
    of a single configuration.
    It will raise an exception if the atoms instance is changed.
    """

    name = "unknown"

    def __init__(
        self,
        atoms,
        dtype=float,
        **results,
    ):
        """Save the properties for the given configuration."""
        super().__init__()
        self.results = {}
        # Save the properties
        for prop, value in results.items():
            if value is None:
                continue
            elif isinstance(value, (float, int)):
                self.results[prop] = value
            else:
                self.results[prop] = array(value, dtype=dtype)
        # Save the configuration
        self.atoms = atoms.copy()

    def __str__(self):
        tokens = []
        for key, val in sorted(self.results.items()):
            if isscalar(val):
                txt = "{}={}".format(key, val)
            else:
                txt = "{}=...".format(key)
            tokens.append(txt)
        return "{}({})".format(self.__class__.__name__, ", ".join(tokens))

    def get_property(self, name, atoms=None, allow_calculation=True):
        if atoms is None:
            atoms = self.atoms
        # Raise an error if the property does not exist or it has changed
        if name not in self.results or self.check_state(atoms):
            if allow_calculation:
                raise PropertyNotImplementedError(
                    'The property "{0}" is not available.'.format(name)
                )
            return None
        # Return the property
        result = self.results[name]
        if isinstance(result, (ndarray, list)):
            result = result.copy()
        return result

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
