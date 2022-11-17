import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from copy import deepcopy

class MLCalculator(Calculator):

    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True

    def __init__(self,mlmodel=None,calculate_uncertainty=True):
        " A ML calculator object applicable in ASE "
        Calculator.__init__(self)
        if mlmodel is None:
            from .mlmodel import MLModel
            mlmodel=MLModel(model=None,database=None,optimize=True,optimize_kwargs={},baseline=None)
        self.mlmodel=deepcopy(mlmodel)
        self.calculate_uncertainty=calculate_uncertainty

    def get_uncertainty(self):
        " Get the calculated uncertainty "
        try:
            return self.results['uncertainty']
        except:
            return None

    def calculate(self,atoms=None,properties=['energy','forces','uncertainty'],system_changes=all_changes):
        """ Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()* and uncertainties using
        *atoms.calc.get_uncertainty()*.
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        if 'forces' in properties:
            # Obtain energy and forces for the given geometry from predictions:
            energy,forces,uncertainty=self.mlmodel.calculate(atoms,get_variance=self.calculate_uncertainty,get_forces=True)
            # Results:
            self.results['energy']=energy
            self.results['forces']=forces
            self.results['uncertainty']=uncertainty
        else:
            # Obtain energy for the given geometry from predictions:
            energy,uncertainty=self.mlmodel.calculate(atoms,get_variance=self.calculate_uncertainty,get_forces=False)
            # Results:
            self.results['energy']=energy
            self.results['uncertainty']=uncertainty