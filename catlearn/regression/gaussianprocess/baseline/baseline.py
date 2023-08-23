import numpy as np
from ase.calculators.calculator import Calculator, all_changes

class Baseline_calculator(Calculator):
    implemented_properties = ['energy','forces']
    nolabel = True
    
    def __init__(self,mic=True,reduce_dimensions=True,**kwargs):
        """ A baseline calculator for ASE atoms object. 
        It uses a flat baseline with zero energy and forces.    
        """
        Calculator.__init__(self)
        self.mic=mic
        self.reduce_dimensions=reduce_dimensions
    
    def calculate(self,atoms=None,properties=['energy','forces'],system_changes=all_changes):
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
        return 0.0,np.zeros((len(atoms),3))
    
    def get_constrains(self,atoms):
        " Get the indicies of the atoms that does not have fixed constrains "
        not_masked=list(range(len(atoms)))
        if not self.reduce_dimensions:
            return not_masked
        constraints=atoms.constraints
        if len(constraints)>0:
            from ase.constraints import FixAtoms
            index_mask=np.array([c.get_indices() for c in constraints if isinstance(c,FixAtoms)]).flatten()
            index_mask=sorted(list(set(index_mask)))
            return [i for i in not_masked if i not in index_mask]
        return not_masked
    
    def copy(self):
        " Copy the calculator. "
        return self.__class__(mic=self.mic,reduce_dimensions=self.reduce_dimensions)
