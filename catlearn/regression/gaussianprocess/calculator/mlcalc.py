import numpy as np
from ase.calculators.calculator import Calculator, all_changes

class MLCalculator(Calculator):

    implemented_properties = ['energy','forces','uncertainty','force uncertainty']
    nolabel = True

    def __init__(self,mlmodel=None,calculate_uncertainty=True,calculate_forces=True,**kwargs):
        " A ML calculator object applicable in ASE "
        Calculator.__init__(self)
        if mlmodel is None:
            from .mlmodel import MLModel
            mlmodel=MLModel(model=None,database=None,optimize=True,optimize_kwargs={},baseline=None)
        self.set_parameters(mlmodel=mlmodel,calculate_uncertainty=calculate_uncertainty,calculate_forces=calculate_forces)

    def get_uncertainty(self):
        " Get the calculated uncertainty "
        return self.results['uncertainty']

    def get_force_uncertainty(self):
        " Get the calculated uncertainty of the forces "
        return self.results['force uncertainty']

    def calculate(self,atoms=None,properties=['energy','forces','uncertainty','force uncertainty'],system_changes=all_changes):
        """ Calculate the energy, forces and uncertainty on the energies for a given Atoms structure. 
            Predicted potential energy can be obtained by *atoms.get_potential_energy()*, 
            predicted forces using *atoms.get_forces()*, 
            uncertainty of the energy using *atoms.calc.get_uncertainty()*, 
            and uncertainties of the forces using *atoms.calc.get_force_uncertainty()*.
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Obtain energy and forces for the given geometry from predictions:
        self.results['energy'],self.results['forces'],self.results['uncertainty'],self.results['force uncertainty']=self.mlmodel.calculate(atoms,get_variance=self.calculate_uncertainty,get_forces=self.calculate_forces)

    def add_training(self,atoms_list,**kwarg):
        " Add training ase Atoms data to the ML model. "
        self.mlmodel.add_training(atoms_list,**kwarg)
        return self
    
    def train_model(self,verbose=False,**kwarg):
        " Train the ML model. "
        self.mlmodel.train_model(verbose=verbose,**kwarg)
        return self
    
    def get_training_set_size(self):
        " Get the number of atoms objects in the ML model. "
        return self.mlmodel.get_training_set_size()

    def save_data(self,trajectory='data.traj',**kwarg):
        " Save the ASE atoms data to a trajectory. "
        self.mlmodel.save_data(trajectory=trajectory,**kwarg)
        return self
    
    def set_parameters(self,mlmodel=None,calculate_uncertainty=None,calculate_forces=None,**kwargs):
        if mlmodel is not None:
            self.mlmodel=mlmodel.copy()
        if calculate_uncertainty is not None:
            self.calculate_uncertainty=calculate_uncertainty
        if calculate_forces is not None:
            self.calculate_forces=calculate_forces
        return self
    
    def copy(self):
        " Copy the calculator. "
        clone=self.__class__(mlmodel=self.mlmodel,
                             calculate_uncertainty=self.calculate_uncertainty,
                             calculate_forces=self.calculate_forces)
        if 'results' in self.__dict__.keys():
            clone.results=self.results.copy()
        return clone
    
    def __deepcopy__(self,memo):
        " Do a deepcopy by using the copy function to make a new object. "
        return self.copy()
    
    def __repr__(self):
        return "MLCalculator(mlmodel={},calculate_uncertainty={},calculate_forces={})".format(self.mlmodel,self.calculate_uncertainty,self.calculate_forces)
