import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from copy import deepcopy

class MLCalculator(Calculator):

    implemented_properties = ['energy','forces','uncertainty','force uncertainty']
    nolabel = True

    def __init__(self,mlmodel=None,calculate_uncertainty=True,calculate_forces=True):
        " A ML calculator object applicable in ASE "
        Calculator.__init__(self)
        if mlmodel is None:
            from .mlmodel import MLModel
            mlmodel=MLModel(model=None,database=None,optimize=True,optimize_kwargs={},baseline=None)
        self.mlmodel=deepcopy(mlmodel)
        self.calculate_uncertainty=calculate_uncertainty
        self.calculate_forces=calculate_forces

    def get_uncertainty(self):
        " Get the calculated uncertainty "
        return self.results['uncertainty']

    def get_force_uncertainty(self):
        " Get the calculated uncertainty of the forces "
        return self.results['force uncertainty']

    def calculate(self,atoms=None,properties=['energy','forces','uncertainty','force uncertainty'],system_changes=all_changes):
        """ Calculate the energy, forces and uncertainty on the energies for a given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using *atoms.get_forces()* and uncertainties using *atoms.calc.get_uncertainty()*.
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Obtain energy and forces for the given geometry from predictions:
        self.results['energy'],self.results['forces'],self.results['uncertainty'],self.results['force uncertainty']=self.mlmodel.calculate(atoms,get_variance=self.calculate_uncertainty,get_forces=self.calculate_forces)
