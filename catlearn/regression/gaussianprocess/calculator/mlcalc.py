import numpy as np
from ase.calculators.calculator import Calculator, all_changes

class MLCalculator(Calculator):

    implemented_properties=['energy','forces','uncertainty','force uncertainty']
    nolabel=True

    def __init__(self,mlmodel=None,calculate_uncertainty=True,calculate_forces=True,**kwargs):
        """
        ML calculator object applicable as an ASE calculator.

        Parameters:
            mlmodel : MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model, and add_training. 
            calculate_uncertainty: bool
                Whether to calculate the uncertainty prediction.
            calculate_forces: bool
                Whether to calculate the forces.
        """
        super().__init__()
        if mlmodel is None:
            from .mlmodel import MLModel
            mlmodel=MLModel(model=None,database=None,baseline=None,optimize=True)
        self.update_arguments(mlmodel=mlmodel,
                              calculate_uncertainty=calculate_uncertainty,
                              calculate_forces=calculate_forces,
                              **kwargs)

    def get_uncertainty(self,**kwargs):
        """
        Get the predicted uncertainty of the energy.

        Returns:
            self.results['uncertainty'] : float
                The predicted uncertainty of the energy.
        """
        return self.results['uncertainty']

    def get_force_uncertainty(self,**kwargs):
        """
        Get the predicted uncertainty of the forces.

        Returns:
            self.results['uncertainty'] : (Nat,3) array
                The predicted uncertainty of the forces.
        """
        return self.results['force uncertainty'].copy()
    
    def add_training(self,atoms_list,**kwarg):
        """
        Add training data as ASE Atoms to the ML model.

        Parameters:
            atoms_list : list or ASE Atoms
                A list of or a single ASE Atoms with calculated energies and forces.

        Returns:
            self: The updated object itself.
        """
        " Add training ase Atoms data to the ML model. "
        self.mlmodel.add_training(atoms_list,**kwarg)
        return self
    
    def train_model(self,**kwarg):
        """ 
        Train the ML model and optimize its hyperparameters if it is chosen. 

        Returns:
            self: The updated object itself.
        """
        self.mlmodel.train_model(**kwarg)
        return self

    def save_data(self,trajectory='data.traj',**kwarg):
        """
        Save the ASE Atoms data to a trajectory.

        Parameters:
            trajectory : str
                The name of the trajectory file where the data is saved.

        Returns:
            self: The updated object itself.
        """
        self.mlmodel.save_data(trajectory=trajectory,**kwarg)
        return self
    
    def get_training_set_size(self):
        """
        Get the number of atoms objects in the ML model.

        Returns:
            int: The number of atoms objects in the ML model.
        """
        return self.mlmodel.get_training_set_size()

    def calculate(self,atoms=None,properties=['energy','forces','uncertainty','force uncertainty'],system_changes=all_changes):
        """ 
        Calculate the prediction energy, forces, and uncertainties of the energies and forces for a given ASE Atoms structure. 
        Predicted potential energy can be obtained by *atoms.get_potential_energy()*, 
        predicted forces using *atoms.get_forces()*, 
        uncertainty of the energy using *atoms.calc.get_uncertainty()*, 
        and uncertainties of the forces using *atoms.calc.get_force_uncertainty()*.

        Returns:
            self.results : dict 
                A dictionary with all the calculated properties.
        """
        # Atoms object.
        super().calculate(atoms,properties,system_changes)
        # Predict energy, forces and uncertainties for the given geometry 
        energy,forces,unc,force_unc=self.model_prediction(atoms)
        self.results['energy']=energy
        self.results['forces']=forces
        self.results['uncertainty']=unc
        self.results['force uncertainty']=force_unc
        return self.results
    
    def update_arguments(self,mlmodel=None,calculate_uncertainty=None,calculate_forces=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters:
            mlmodel : MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model, and add_training. 
            calculate_uncertainty: bool
                Whether to calculate the uncertainty prediction.
            calculate_forces: bool
                Whether to calculate the forces.
                
        Returns:
            self: The updated object itself.
        """
        if mlmodel is not None:
            self.mlmodel=mlmodel.copy()
        if calculate_uncertainty is not None:
            self.calculate_uncertainty=calculate_uncertainty
        if calculate_forces is not None:
            self.calculate_forces=calculate_forces
        # Empty the results
        self.reset()
        return self

    def model_prediction(self,atoms,**kwargs):
        " Predict energy, forces and uncertainties for the given geometry with the ML model. "
        energy,forces,unc,force_unc=self.mlmodel.calculate(atoms,
                                                           get_variance=self.calculate_uncertainty,
                                                           get_forces=self.calculate_forces)
        return energy,forces,unc,force_unc
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mlmodel=self.mlmodel,
                        calculate_uncertainty=True,
                        calculate_forces=True)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs

    def copy(self):
        " Copy the object. "
        # Get all arguments
        arg_kwargs,constant_kwargs,object_kwargs=self.get_arguments()
        # Make a clone
        clone=self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key,value in constant_kwargs.items():
                clone.__dict__[key]=value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key,value in object_kwargs.items():
                clone.__dict__[key]=value.copy()
        return clone
    
    def __repr__(self):
        arg_kwargs=self.get_arguments()[0]
        str_kwargs=",".join([f"{key}={value}" for key,value in arg_kwargs.items()])
        return "{}({})".format(self.__class__.__name__,str_kwargs)
    
    def __deepcopy__(self,memo):
        " Do a deepcopy by using the copy function to make a new object. "
        return self.copy()
    