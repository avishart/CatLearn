from .mlcalc import MLCalculator
from ..fingerprint.geometry import sine_activation
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
        max_unc=None,
        max_unc_scale=0.95,
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
            max_unc: float (optional)
                The maximum uncertainty value that can be added to the energy.
                If the uncertainty is larger than the max_unc_scale times this
                value, the cutoff is activated to limit the uncertainty.
            max_unc_scale: float (optional)
                The scale of the maximum uncertainty value to start the cutoff.
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
            max_unc=max_unc,
            max_unc_scale=max_unc_scale,
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
            # Get the uncertainty and its derivatives
            unc = self.results["uncertainty"]
            if get_forces:
                unc_deriv = self.results["uncertainty derivatives"]
            else:
                unc_deriv = None
            # Limit the uncertainty to the maximum uncertainty
            if self.max_unc is not None and unc > self.max_unc_start:
                unc, unc_deriv = self.max_unc_activation(
                    unc,
                    unc_deriv=unc_deriv,
                    use_derivatives=get_forces,
                )
            # Add the uncertainty to the energy and forces
            self.results["energy"] += self.kappa * unc
            if get_forces:
                self.results["forces"] -= self.kappa * unc_deriv
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
        max_unc=None,
        max_unc_scale=None,
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
            max_unc: float (optional)
                The maximum uncertainty value that can be added to the energy.
                If the uncertainty is larger than the max_unc_scale times this
                value, the cutoff is activated to limit the uncertainty.
            max_unc_scale: float (optional)
                The scale of the maximum uncertainty value to start the cutoff.

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
        elif not hasattr(self, "kappa"):
            self.set_kappa(0.0)
        # Set the maximum uncertainty value
        if max_unc is not None:
            self.max_unc = abs(float(max_unc))
        elif not hasattr(self, "max_unc"):
            self.max_unc = None
        if max_unc_scale is not None:
            self.max_unc_scale = float(max_unc_scale)
            if self.max_unc_scale > 1.0:
                raise ValueError(
                    "max_unc_scale must be less than or equal to 1.0"
                )
        if self.max_unc is not None:
            self.max_unc_start = self.max_unc_scale * self.max_unc
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

    def max_unc_activation(self, unc, unc_deriv=None, use_derivatives=False):
        # Calculate the activation function
        fc, gc = sine_activation(
            unc,
            use_derivatives=use_derivatives,
            xs_activation=self.max_unc_start,
            xe_activation=self.max_unc,
        )
        # Calculate the derivative of the uncertainty
        if use_derivatives:
            unc_deriv = unc_deriv * (1.0 - fc)
            unc_deriv += gc * (self.max_unc_start - unc)
        # Apply the activation function to the uncertainty
        unc = (unc * (1.0 - fc)) + (self.max_unc * fc)
        return unc, unc_deriv

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
            max_unc=self.max_unc,
            max_unc_scale=self.max_unc_scale,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
