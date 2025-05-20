from .mlcalc import MLCalculator
from ase.calculators.calculator import Calculator, all_changes


class BOCalculator(MLCalculator):
    """
    The machine learning calculator object applicable as an ASE calculator for
    ASE Atoms instance.
    This uses an acquisition function as the energy and forces.
    E = E_pred + kappa * sigma
    Therefore, it is Bayesian optimization calculator object.
    """

    # Define the properties available in this calculator
    implemented_properties = [
        "energy",
        "forces",
        "uncertainty",
        "force uncertainties",
        "predicted energy",
        "predicted forces",
        "uncertainty derivatives",
    ]
    nolabel = True

    def __init__(
        self,
        mlmodel=None,
        calc_forces=True,
        calc_unc=True,
        calc_force_unc=False,
        calc_unc_deriv=True,
        calc_kwargs={},
        round_pred=None,
        kappa=2.0,
        **kwargs,
    ):
        """
        Initialize the ML calculator.

        Parameters:
            mlmodel: MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model,
                and add_training.
            calc_forces: bool
                Whether to calculate the forces.
            calc_unc: bool
                Whether to calculate
                the uncertainty prediction of the energy.
            calc_force_unc: bool
                Whether to calculate
                the uncertainties of the force predictions.
            calc_unc_deriv: bool
                Whether to calculate
                the derivatives of the uncertainty of the energy.
            calc_kwargs: dict
                A dictionary with kwargs for
                the parent calculator class object.
            round_pred: int (optional)
                The number of decimals to round the preditions to.
                If None, the predictions are not rounded.
            kappa: float
                The weight of the uncertainty relative to the energy.
                If kappa>0, the uncertainty is added to the predicted energy.
        """
        super().__init__(
            mlmodel=mlmodel,
            calc_forces=calc_forces,
            calc_unc=calc_unc,
            calc_force_unc=calc_force_unc,
            calc_unc_deriv=calc_unc_deriv,
            calc_kwargs=calc_kwargs,
            round_pred=round_pred,
            kappa=kappa,
            **kwargs,
        )

    def get_predicted_energy(self, atoms=None, **kwargs):
        """
        Get the predicted energy without the uncertainty.

        Parameters:
            atoms: ASE Atoms (optional)
                The ASE Atoms instance which is used
                if the uncertainty is not stored.

        Returns:
            float: The predicted energy.
        """
        return self.get_property("predicted energy", atoms=atoms)

    def get_predicted_forces(self, atoms=None, **kwargs):
        """
        Get the predicted forces without the derivatives of the uncertainty.

        Parameters:
            atoms: ASE Atoms (optional)
                The ASE Atoms instance which is used
                if the uncertainty is not stored.

        Returns:
            (Nat,3) array: The predicted forces.
        """
        return self.get_property("predicted forces", atoms=atoms)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """
        Calculate the prediction energy, forces, and uncertainties of
        the energies and forces for a given ASE Atoms structure.
        Predicted potential energy can be obtained
        by *atoms.get_potential_energy()*,
        predicted forces using *atoms.get_forces()*,
        uncertainty of the energy using *atoms.calc.get_uncertainty(atoms)*,
        uncertainties of the forces
        using *atoms.calc.get_force_uncertainty(atoms)*,
        derivatives of uncertainty
        using *atoms.calc.get_uncertainty_derivatives(atoms)*,
        predicted energy using *atoms.calc.get_predicted_energy(atoms)*,
        and predicted forces using *atoms.calc.get_predicted_forces(atoms)*.

        Returns:
            self.results: dict
                A dictionary with all the calculated properties.
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Get the arguments for calculating the requested properties
        (
            get_forces,
            get_uncertainty,
            get_force_uncertainties,
            get_unc_derivatives,
        ) = self.get_property_arguments(properties)
        # Get predict energy, forces and uncertainties for the given geometry
        results = self.model_prediction(
            atoms,
            get_forces=get_forces,
            get_uncertainty=get_uncertainty,
            get_force_uncertainties=get_force_uncertainties,
            get_unc_derivatives=get_unc_derivatives,
        )
        # Store the properties that are implemented
        self.store_properties(results)
        # Save the predicted properties
        self.modify_results_bo(
            get_forces=get_forces,
        )
        return self.results

    def modify_results_bo(
        self,
        get_forces,
        **kwargs,
    ):
        """
        Modify the results of the Bayesian optimization calculator.
        """
        # Save the predicted properties
        self.results["predicted energy"] = self.results["energy"]
        if get_forces:
            self.results["predicted forces"] = self.results["forces"].copy()
        # Calculate the acquisition function and its derivative
        if self.kappa != 0.0:
            self.results["energy"] += self.kappa * self.results["uncertainty"]
            if get_forces:
                self.results["forces"] -= (
                    self.kappa * self.results["uncertainty derivatives"]
                )
        return self.results

    def update_arguments(
        self,
        mlmodel=None,
        calc_forces=None,
        calc_unc=None,
        calc_force_unc=None,
        calc_unc_deriv=None,
        calc_kwargs=None,
        round_pred=None,
        kappa=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            mlmodel: MLModel class object
                Machine Learning model used for ASE Atoms and calculator.
                The object must have the functions: calculate, train_model,
                and add_training.
            calc_forces: bool
                Whether to calculate the forces.
            calc_unc: bool
                Whether to calculate
                the uncertainty prediction of the energy.
            calc_force_unc: bool
                Whether to calculate
                the uncertainties of the force predictions.
            calc_unc_deriv: bool
                Whether to calculate
                the derivatives of the uncertainty of the energy.
            calc_kwargs: dict
                A dictionary with kwargs for
                the parent calculator class object.
            round_pred: int (optional)
                The number of decimals to round the preditions to.
                If None, the predictions are not rounded.
            kappa: float
                The weight of the uncertainty relative to the energy.

        Returns:
            self: The updated object itself.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            mlmodel=mlmodel,
            calc_forces=calc_forces,
            calc_unc=calc_unc,
            calc_force_unc=calc_force_unc,
            calc_unc_deriv=calc_unc_deriv,
            calc_kwargs=calc_kwargs,
            round_pred=round_pred,
        )
        # Set the kappa value
        if kappa is not None:
            self.set_kappa(kappa)
        return self

    def set_kappa(self, kappa, **kwargs):
        """
        Set the kappa value.
        The kappa value is used to calculate the acquisition function.

        Parameters:
            kappa: float
                The weight of the uncertainty relative to the energy.
        """
        self.kappa = float(kappa)
        self.reset()
        return self

    def get_property_arguments(self, properties=[], **kwargs):
        # Check if the forces must be predicted
        if (
            self.calc_forces
            or "forces" in properties
            or "predicted forces" in properties
        ):
            get_forces = True
        else:
            get_forces = False
        # Check if the uncertainty must be predicted
        if self.calc_unc or "uncertainty" in properties or self.kappa != 0.0:
            get_uncertainty = True
        else:
            get_uncertainty = False
        # Check if the force uncertainties must be predicted
        if self.calc_force_unc or "force uncertainties" in properties:
            get_force_uncertainties = True
        else:
            get_force_uncertainties = False
        # Check if the derivatives of the uncertainty must be predicted
        if (
            self.calc_unc_deriv
            or "uncertainty derivatives" in properties
            or (get_forces and self.kappa != 0.0)
        ):
            get_unc_derivatives = True
        else:
            get_unc_derivatives = False
        return (
            get_forces,
            get_uncertainty,
            get_force_uncertainties,
            get_unc_derivatives,
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            mlmodel=self.mlmodel,
            calc_forces=self.calc_forces,
            calc_unc=self.calc_unc,
            calc_force_unc=self.calc_force_unc,
            calc_unc_deriv=self.calc_unc_deriv,
            calc_kwargs=self.calc_kwargs,
            round_pred=self.round_pred,
            kappa=self.kappa,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
