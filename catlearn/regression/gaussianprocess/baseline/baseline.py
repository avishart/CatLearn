import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

class Baseline_calculator(Calculator):
    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True
    
    def __init__(self):
        """ A baseline calculator for ASE atoms object. 
        It uses a flat baseline with zero energy and forces.    
        """
        Calculator.__init__(self)
        pass
    
    def calculate(self,atoms=None,properties=['energy', 'forces', 'uncertainty'],system_changes=all_changes):
        """
        Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()*.
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Obtain energy and forces for the given structure:
        if 'forces' in properties:
            energy,forces=self.get_energy_forces(atoms)
            self.results['forces']=forces
        else:
            energy=self.get_energy(atoms)
        self.results['energy']=energy
        pass

    def get_energy(self,atoms):
        " Get only the energy. "
        return 0.0

    def get_energy_forces(self,atoms):
        " Get the energy and forces. "
        return np.array([[0.0]*3]*len(atoms))
