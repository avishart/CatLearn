from .mlcalc import MLCalculator
from ase.calculators.calculator import Calculator,all_changes

class BOCalculator(MLCalculator):

    # Define the properties available in this calculator 
    implemented_properties=['energy','forces','uncertainty','force uncertainties','predicted energy','predicted forces','uncertainty derivatives']
    nolabel=True

    def __init__(self,mlmodel=None,calculate_forces=True,calculate_uncertainty=True,calculate_force_uncertainties=False,calculate_unc_derivatives=True,calculator_kwargs={},kappa=2.0,**kwargs):
        """
        Bayesian optimization calculator object applicable as an ASE calculator.

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
            kappa : float
                The weight of the uncertainty relative to the energy.
                If kappa>0, the uncertainty is added to the predicted energy.
        """
        super().__init__(mlmodel=mlmodel,
                         calculate_forces=calculate_forces,
                         calculate_uncertainty=calculate_uncertainty,
                         calculate_force_uncertainties=calculate_force_uncertainties,
                         calculate_unc_derivatives=calculate_unc_derivatives,
                         calculator_kwargs=calculator_kwargs,
                         kappa=kappa,
                         **kwargs)

    def get_predicted_energy(self,atoms=None,**kwargs):
        """
        Get the predicted energy without the uncertainty.

        Parameters:
            atoms : ASE Atoms (optional)
                The ASE Atoms object which is used if the predicted energy is not stored.

        Returns:
            float: The predicted energy.
        """
        return self.get_property('predicted energy',atoms=atoms)

    def get_predicted_forces(self,atoms=None,**kwargs):
        """
        Get the predicted forces without the derivatives of the uncertainty.

        Parameters:
            atoms : ASE Atoms (optional)
                The ASE Atoms object which is used if the predicted forces is not stored.

        Returns:
            (Nat,3) array: The predicted forces.
        """
        return self.get_property('predicted forces',atoms=atoms)

    def calculate(self,atoms=None,properties=['energy','forces'],system_changes=all_changes):
        """ 
        Calculate the prediction energy, forces, and uncertainties of the energies and forces for a given ASE Atoms structure. 
        Predicted potential energy can be obtained by *atoms.get_potential_energy()*, 
        predicted forces using *atoms.get_forces()*, 
        uncertainty of the energy using *atoms.calc.get_uncertainty(atoms)*, 
        uncertainties of the forces using *atoms.calc.get_force_uncertainty(atoms)*,
        derivatives of uncertainty using *atoms.calc.get_uncertainty_derivatives(atoms)*,
        predicted energy using *atoms.calc.get_predicted_energy(atoms)*,
        and predicted forces using *atoms.calc.get_predicted_forces(atoms)*.

        Returns:
            self.results : dict 
                A dictionary with all the calculated properties.
        """
        # Atoms object.
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
        # Save the predicted properties
        self.results['predicted energy']=results['energy']
        if get_forces:
            self.results['predicted forces']=results['forces'].copy()
        # Calculate the acquisition function and its derivative
        if self.kappa!=0.0:
            self.results['energy']=results['energy']+self.kappa*results['uncertainty']
            if get_forces:
                self.results['forces']=results['forces']-self.kappa*results['uncertainty derivatives']
        return self.results

    def update_arguments(self,mlmodel=None,calculate_forces=None,calculate_uncertainty=None,calculate_force_uncertainties=None,calculate_unc_derivatives=None,calculator_kwargs=None,kappa=None,**kwargs):
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
            kappa : float
                The weight of the uncertainty relative to the energy.
                
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
        if kappa is not None:
            self.kappa=float(kappa)
        # Empty the results
        self.reset()
        return self
    
    def get_property_arguments(self,properties=[],**kwargs):
        # Check if the forces must be predicted
        if self.calculate_forces or 'forces' in properties or 'predicted forces' in properties:
            get_forces=True
        else:
            get_forces=False
        # Check if the uncertainty must be predicted
        if self.calculate_uncertainty or 'uncertainty' in properties or self.kappa!=0.0:
            get_uncertainty=True
        else:
            get_uncertainty=False
        # Check if the force uncertainties must be predicted
        if self.calculate_force_uncertainties or 'force uncertainties' in properties:
            get_force_uncertainties=True
        else:
            get_force_uncertainties=False
        # Check if the derivatives of the uncertainty must be predicted
        if self.calculate_unc_derivatives or 'uncertainty derivatives' in properties or (get_forces and self.kapa!=0.0):
            get_unc_derivatives=True
        else:
            get_unc_derivatives=False
        return get_forces,get_uncertainty,get_force_uncertainties,get_unc_derivatives

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mlmodel=self.mlmodel,
                        calculate_forces=self.calculate_forces,
                        calculate_uncertainty=self.calculate_uncertainty,
                        calculate_force_uncertainties=self.calculate_force_uncertainties,
                        calculate_unc_derivatives=self.calculate_unc_derivatives,
                        calculator_kwargs=self.calculator_kwargs,
                        kappa=self.kappa)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
