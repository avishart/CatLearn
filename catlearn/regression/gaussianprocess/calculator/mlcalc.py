import numpy as np
from ase.calculators.calculator import Calculator, all_changes

class MLCalculator(Calculator):

    # Define the properties available in this calculator 
    implemented_properties=['energy','forces','uncertainty','force uncertainties','uncertainty derivatives']
    nolabel=True

    def __init__(self,mlmodel=None,calculate_forces=True,calculate_uncertainty=True,calculate_force_uncertainties=False,calculate_unc_derivatives=False,calculator_kwargs={},**kwargs):
        """
        ML calculator object applicable as an ASE calculator.

        Parameters:
            mlmodel : MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model, and add_training. 
            calculate_forces: bool
                Whether to calculate the forces.
            calculate_uncertainty: bool
                Whether to calculate the uncertainty prediction of the energy.
            calculate_force_uncertainties: bool
                Whether to calculate the uncertainties of the force predictions.
            calculate_unc_derivatives: bool
                Whether to calculate the derivatives of the uncertainty of the energy.
            calculator_kwargs : dict
                A dictionary with kwargs for the Parent calculator class object.
        """
        # Inherit from the Calculator object
        Calculator.__init__(self,**calculator_kwargs)
        # Set default mlmodel
        if mlmodel is None:
            from .mlmodel import MLModel
            mlmodel=MLModel(model=None,database=None,baseline=None,optimize=True)
        # Set all the arguments
        self.update_arguments(mlmodel=mlmodel,
                              calculate_forces=calculate_forces,
                              calculate_uncertainty=calculate_uncertainty,
                              calculate_force_uncertainties=calculate_force_uncertainties,
                              calculate_unc_derivatives=calculate_unc_derivatives,
                              calculator_kwargs=calculator_kwargs,
                              **kwargs)

    def get_uncertainty(self,atoms=None,**kwargs):
        """
        Get the predicted uncertainty of the energy.

        Parameters:
            atoms : ASE Atoms (optional)
                The ASE Atoms instance which is used if the uncertainty is not stored.

        Returns:
            float: The predicted uncertainty of the energy.
        """
        return self.get_property('uncertainty',atoms=atoms)

    def get_force_uncertainty(self,atoms=None,**kwargs):
        """
        Get the predicted uncertainty of the forces.

        Parameters:
            atoms : ASE Atoms (optional)
                The ASE Atoms instance which is used if the force uncertainties are not stored.

        Returns:
            (Nat,3) array: The predicted uncertainty of the forces.
        """
        return self.get_property('force uncertainties',atoms=atoms)
    
    def get_uncertainty_derivatives(self,atoms=None,**kwargs):
        """
        Get the derivatives of the uncertainty of the energy.

        Parameters:
            atoms : ASE Atoms (optional)
                The ASE Atoms instance which is used if the derivatives of the uncertainty are not stored.

        Returns:
            (Nat,3) array: The predicted uncertainty of the forces.
        """
        return self.get_property('uncertainty derivatives',atoms=atoms)
    
    def set_atoms(self,atoms,**kwargs):
        """ 
        Save the ASE Atoms instance in the calculator.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms instance that are saved.

        Returns:
            self: The updated object itself.
        """
        if self.check_state(atoms):
            self.reset()
            self.atoms=atoms.copy()
        return self
    
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
    
    def is_in_database(self,atoms,**kwargs):
        """ 
        Check if the ASE Atoms is in the database.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms instance with a calculator.

        Returns:
            bool: Whether the ASE Atoms instance is within the database.
        """
        return self.mlmodel.is_in_database(atoms=atoms,**kwargs)
    
    def copy_atoms(self,atoms,**kwargs):
        """
        Copy the atoms object together with the calculated properties.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator that is copied.

        Returns:
            ASE Atoms: The copy of the Atoms object with saved data in the calculator.
        """
        return self.mlmodel.copy_atoms(atoms,**kwargs)
    
    def update_database_arguments(self,point_interest=None,**kwargs):
        """ 
        Update the arguments in the database.

        Parameters:
            point_interest : list
                A list of the points of interest as ASE Atoms instances. 

        Returns:
            self: The updated object itself.
        """
        self.mlmodel.update_database_arguments(point_interest=point_interest,**kwargs)
        return self

    def calculate(self,atoms=None,properties=['energy','forces'],system_changes=all_changes):
        """ 
        Calculate the prediction energy, forces, and uncertainties of the energies and forces for a given ASE Atoms structure. 
        Predicted potential energy can be obtained by *atoms.get_potential_energy()*, 
        predicted forces using *atoms.get_forces()*, 
        uncertainty of the energy using *atoms.calc.get_uncertainty(atoms)*, 
        uncertainties of the forces using *atoms.calc.get_force_uncertainty(atoms)*,
        and derivatives of uncertainty using *atoms.calc.get_uncertainty_derivatives(atoms)*.

        Returns:
            self.results : dict 
                A dictionary with all the calculated properties.
        """
        # Atoms object
        Calculator.calculate(self,atoms,properties,system_changes)
        # Get the arguments for calculating the requested properties
        get_forces,get_uncertainty,get_force_uncertainties,get_unc_derivatives=self.get_property_arguments(properties)
        # Get predict energy, forces and uncertainties for the given geometry 
        results=self.model_prediction(atoms,
                                      get_forces=get_forces,
                                      get_uncertainty=get_uncertainty,
                                      get_force_uncertainties=get_force_uncertainties,
                                      get_unc_derivatives=get_unc_derivatives)
        # Store the properties that are implemented
        for key,value in results.items():
            if key in self.implemented_properties:
                self.results[key]=value
        return self.results
    
    def update_arguments(self,mlmodel=None,calculate_forces=None,calculate_uncertainty=None,calculate_force_uncertainties=None,calculate_unc_derivatives=None,calculator_kwargs=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters:
            mlmodel : MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model, and add_training. 
            calculate_forces: bool
                Whether to calculate the forces.
            calculate_uncertainty: bool
                Whether to calculate the uncertainty prediction.
            calculate_force_uncertainties: bool
                Whether to calculate the uncertainties of the force predictions.
            calculate_unc_derivatives: bool
                Whether to calculate the derivatives of the uncertainty of the energy.
            calculator_kwargs : dict
                A dictionary with kwargs for the Parent calculator class object.
                
        Returns:
            self: The updated object itself.
        """
        if mlmodel is not None:
            self.mlmodel=mlmodel.copy()
        if calculate_forces is not None:
            self.calculate_forces=calculate_forces
        if calculate_uncertainty is not None:
            self.calculate_uncertainty=calculate_uncertainty
        if calculate_force_uncertainties is not None:
            self.calculate_force_uncertainties=calculate_force_uncertainties
        if calculate_unc_derivatives is not None:
            self.calculate_unc_derivatives=calculate_unc_derivatives
        if calculator_kwargs is not None:
            self.calculator_kwargs=calculator_kwargs.copy()
        # Empty the results
        self.reset()
        return self
    
    def get_property_arguments(self,properties=[],**kwargs):
        " Get the arguments that ensure calculations of the properties requested. "
        # Check if the forces must be predicted
        if self.calculate_forces or 'forces' in properties:
            get_forces=True
        else:
            get_forces=False
        # Check if the uncertainty must be predicted
        if self.calculate_uncertainty or 'uncertainty' in properties:
            get_uncertainty=True
        else:
            get_uncertainty=False
        # Check if the force uncertainties must be predicted
        if self.calculate_force_uncertainties or 'force uncertainties' in properties:
            get_force_uncertainties=True
        else:
            get_force_uncertainties=False
        # Check if the derivatives of the uncertainty must be predicted
        if self.calculate_unc_derivatives or 'uncertainty derivatives' in properties:
            get_unc_derivatives=True
        else:
            get_unc_derivatives=False
        return get_forces,get_uncertainty,get_force_uncertainties,get_unc_derivatives

    def model_prediction(self,atoms,get_forces=False,get_uncertainty=False,get_force_uncertainties=False,get_unc_derivatives=False,**kwargs):
        " Predict energy, forces and uncertainties for the given geometry with the ML model. "
        results=self.mlmodel.calculate(atoms,
                                       get_forces=get_forces,
                                       get_uncertainty=get_uncertainty,
                                       get_force_uncertainties=get_force_uncertainties,
                                       get_unc_derivatives=get_unc_derivatives,
                                       **kwargs)
        return results
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mlmodel=self.mlmodel,
                        calculate_forces=self.calculate_forces,
                        calculate_uncertainty=self.calculate_uncertainty,
                        calculate_force_uncertainties=self.calculate_force_uncertainties,
                        calculate_unc_derivatives=self.calculate_unc_derivatives,
                        calculator_kwargs=self.calculator_kwargs)
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
    