from numpy import einsum, sqrt
from ase import Atoms
from ..regression.gp.calculator.copy_atoms import copy_atoms


class Structure(Atoms):
    def __init__(self, atoms, *args, **kwargs):
        self.atoms = atoms
        self.__dict__.update(atoms.__dict__)
        if atoms.calc is not None and len(atoms.calc.results):
            self.store_results()
        else:
            self.reset()

    def set_positions(self, *args, **kwargs):
        self.atoms.set_positions(*args, **kwargs)
        self.reset()
        return

    def set_scaled_positions(self, *args, **kwargs):
        self.atoms.set_scaled_positions(*args, **kwargs)
        self.reset()
        return

    def set_cell(self, *args, **kwargs):
        self.atoms.set_cell(*args, **kwargs)
        self.reset()
        return

    def set_pbc(self, *args, **kwargs):
        self.atoms.set_pbc(*args, **kwargs)
        self.reset()
        return

    def set_initial_charges(self, *args, **kwargs):
        self.atoms.set_initial_charges(*args, **kwargs)
        self.reset()
        return

    def set_initial_magnetic_moments(self, *args, **kwargs):
        self.atoms.set_initial_magnetic_moments(*args, **kwargs)
        self.reset()
        return

    def set_momenta(self, *args, **kwargs):
        self.atoms.set_momenta(*args, **kwargs)
        self.reset()
        return

    def set_velocities(self, *args, **kwargs):
        self.atoms.set_velocities(*args, **kwargs)
        self.reset()
        return

    def get_property(self, name, allow_calculation=True, **kwargs):
        """
        Get or calculate the requested property.

        Parameters:
            name : str
                The name of the requested property.
            allow_calculation : bool
                Whether the property is allowed to be calculated.

        Returns:
            float or list: The requested property.
        """
        if self.is_saved:
            if name in self.results:
                output = self.atoms_saved.calc.get_property(
                    name,
                    atoms=self.atoms_saved,
                    allow_calculation=True,
                    **kwargs,
                )
                return output
        output = self.atoms.calc.get_property(
            name,
            atoms=self.atoms,
            allow_calculation=allow_calculation,
            **kwargs,
        )
        self.store_results()
        return output

    def get_forces(self, *args, **kwargs):
        if self.is_saved:
            if "force" in self.results:
                return self.atoms_saved.get_forces(*args, **kwargs)
        forces = self.atoms.get_forces(*args, **kwargs)
        self.store_results()
        return forces

    def get_potential_energy(self, *args, **kwargs):
        if self.is_saved:
            if "energy" in self.results:
                return self.atoms_saved.get_potential_energy(*args, **kwargs)
        energy = self.atoms.get_potential_energy(*args, **kwargs)
        self.store_results()
        return energy

    def get_x(self):
        return self.get_positions().ravel()

    def set_x(self, x):
        self.set_positions(x.reshape(-1, 3))

    def get_gradient(self):
        return self.get_forces().ravel()

    def get_value(self, *args, **kwargs):
        return self.get_potential_energy(*args, **kwargs)

    def gradient_norm(self, gradient):
        forces = gradient.reshape(-1, 3)
        return sqrt(einsum("ij,ij->i", forces, forces)).max()

    def get_uncertainty(self, *args, **kwargs):
        if self.is_saved:
            if "uncertainty" in self.results:
                unc = self.atoms_saved.calc.get_uncertainty(
                    self.atoms_saved,
                    *args,
                    **kwargs,
                )
                return unc
        unc = self.atoms.calc.get_uncertainty(
            self.atoms,
            *args,
            **kwargs,
        )
        self.store_results()
        return unc

    def converged(self, forces, fmax):
        return sqrt(einsum("ij,ij->i", forces, forces)).max() < fmax

    def is_neb(self):
        return False

    def __ase_optimizable__(self):
        return self

    def set_calculator(self, calc, copy_calc=False, **kwargs):
        if copy_calc:
            self.atoms.calc = calc.copy()
        else:
            self.atoms.calc = calc
        self.reset()
        return

    @property
    def calc(self):
        """
        The calculator objects.
        """
        if self.is_saved:
            return self.atoms_saved.calc
        return self.atoms.calc

    @calc.setter
    def calc(self, calc):
        return self.set_calculator(calc)

    def copy(self):
        return self.atoms.copy()

    def get_structure(self):
        return self.atoms

    def get_atoms(self):
        return self.get_structure()

    def get_saved_structure(self):
        return self.atoms_saved

    def reset(self):
        self.atoms_saved = self.atoms.copy()
        self.results = {}
        self.is_saved = False
        return self

    def store_results(self, **kwargs):
        """
        Store the calculated results.
        """
        self.atoms_saved = copy_atoms(self.atoms)
        self.results = self.atoms_saved.calc.results.copy()
        self.is_saved = True
        return self.atoms_saved
