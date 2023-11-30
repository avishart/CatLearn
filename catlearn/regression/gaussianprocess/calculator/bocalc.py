from .mlcalc import MLCalculator
from ase.calculators.calculator import all_changes

class BOCalculator(MLCalculator):

    # Define the properties available in this calculator 
    implemented_properties=['energy','forces','uncertainty','force uncertainties','predicted energy','predicted forces','uncertainty derivatives']
    nolabel=True

    def __init__(self,mlmodel=None,calculate_forces=True,calculate_uncertainty=True,calculate_force_uncertainties=False,kappa=2.0,**kwargs):
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
            kappa : float
                The weight of the uncertainty relative to the energy.
                If kappa>0, the uncertainty is added to the predicted energy.
        """
        super().__init__(mlmodel=mlmodel,
                         calculate_forces=calculate_forces,
                         calculate_uncertainty=calculate_uncertainty,
                         calculate_force_uncertainties=calculate_force_uncertainties,
                         kappa=kappa,
                         **kwargs)

    def get_predicted_energy(self,**kwargs):
        """
        Get the predicted energy without the uncertainty.

        Returns:
            self.results['predicted energy'] : float
                The predicted energy.
        """
        if 'predicted energy' in self.results:
            return self.results['predicted energy']
        return None

    def get_predicted_forces(self,**kwargs):
        """
        Get the predicted forces without the derivatives of the uncertainty.

        Returns:
            self.results['predicted forces'] : (Nat,3) array
                The predicted forces.
        """
        if 'predicted forces' in self.results:
            return self.results['predicted forces']
        return None

    def calculate(self,atoms=None,properties=['energy','forces','uncertainty','force uncertainties','predicted energy','predicted forces','uncertainty derivatives'],system_changes=all_changes):
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
        # Get predict energy, forces and uncertainties for the given geometry 
        results=self.model_prediction(atoms)
        # Store the properties that are implemented
        self.results={key:value for key,value in results.items() if key in self.implemented_properties}
        # Save the predicted properties
        self.results['predicted energy']=results['energy']
        self.results['predicted forces']=results['forces']
        # Calculate the acquisition function and its derivative
        if self.calculate_uncertainty:
            self.results['energy']=results['energy']+self.kappa*results['uncertainty']
            self.results['forces']=results['forces']-self.kappa*results['uncertainty derivatives']
        return self.results

    def update_arguments(self,mlmodel=None,calculate_forces=None,calculate_uncertainty=None,calculate_force_uncertainties=None,kappa=None,**kwargs):
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
        if kappa is not None:
            self.kappa=float(kappa)
        # Empty the results
        self.reset()
        return self

    def model_prediction(self,atoms,**kwargs):
        " Predict energy, forces and uncertainties for the given geometry with the ML model. "
        results=self.mlmodel.calculate(atoms,
                                       get_uncertainty=self.calculate_uncertainty,
                                       get_forces=self.calculate_forces,
                                       get_force_uncertainties=self.calculate_force_uncertainties,
                                       get_unc_derivatives=True,
                                       **kwargs)
        return results

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mlmodel=self.mlmodel,
                        calculate_forces=self.calculate_forces,
                        calculate_uncertainty=self.calculate_uncertainty,
                        calculate_force_uncertainties=self.calculate_force_uncertainties,
                        kappa=self.kappa)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
