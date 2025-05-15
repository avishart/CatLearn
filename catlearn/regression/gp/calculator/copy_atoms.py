from numpy import array, isscalar, ndarray
from ase.calculators.calculator import Calculator, PropertyNotImplementedError


def copy_atoms(atoms, results={}, **kwargs):
    """
    Copy the atoms object together with the calculated properties.

    Parameters:
        atoms : ASE Atoms
            The ASE Atoms object with a calculator that is copied.
        results : dict (optional)
            The properties to be saved in the calculator.
            If not given, the properties are taken from the calculator.

    Returns:
        atoms0 : ASE Atoms
            The copy of the Atoms object with saved data in the calculator.
    """
    # Check if results are given
    if not isinstance(results, dict) or len(results) == 0:
        # Save the properties calculated
        if atoms.calc is not None:
            results = atoms.calc.results.copy()
    # Copy the ASE Atoms object
    atoms0 = atoms.copy()
    # Store the properties in a calculator
    atoms0.calc = StoredDataCalculator(atoms, **results)
    return atoms0


class StoredDataCalculator(Calculator):
    """
    A special calculator that store the data (results)
    of a single configuration.
    It will raise an exception if the atoms object is changed.
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
            atoms : ASE Atoms (optional)
                The ASE Atoms instance which is used
                if the uncertainty is not stored.

        Returns:
            float: The predicted uncertainty of the energy.
        """
        return self.get_property("uncertainty", atoms=atoms)
