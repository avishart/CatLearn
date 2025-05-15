from numpy import inf
from scipy.optimize import OptimizeResult
from .hpfitter import (
    FunctionEvaluation,
    HyperparameterFitter,
    VariableTransformation,
)


class ReducedHyperparameterFitter(HyperparameterFitter):
    def __init__(
        self,
        func,
        optimizer=FunctionEvaluation(jac=False),
        bounds=VariableTransformation(),
        use_update_pdis=False,
        get_prior_mean=False,
        use_stored_sols=False,
        round_hp=None,
        opt_tr_size=50,
        dtype=float,
        **kwargs,
    ):
        """
        Hyperparameter fitter object with an optimizer for optimizing
        the hyperparameters on different given objective functions.
        The optimization of the hyperparameters are only performed when
        the training set size is below a number.

        Parameters:
            func: ObjectiveFunction class
                A class with the objective function used
                to optimize the hyperparameters.
            optimizer: Optimizer class
                A class with the used optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                Most of the global optimizers are using boundary conditions.
                The bounds in this class will be used
                for the optimizer and func.
            use_update_pdis: bool
                Whether to update the prior distributions of
                the hyperparameters with the given boundary conditions.
            get_prior_mean: bool
                Whether to get the parameters of the prior mean
                in the solution.
            use_stored_sols: bool
                Whether to store the solutions.
            round_hp: int (optional)
                The number of decimals to round the hyperparameters to.
                If None, the hyperparameters are not rounded.
            opt_tr_size: int
                The maximum size of the training set before
                the hyperparameters are not optimized.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            func,
            optimizer=optimizer,
            bounds=bounds,
            use_update_pdis=use_update_pdis,
            get_prior_mean=get_prior_mean,
            use_stored_sols=use_stored_sols,
            round_hp=round_hp,
            opt_tr_size=opt_tr_size,
            dtype=dtype,
            **kwargs,
        )

    def fit(self, X, Y, model, hp=None, pdis=None, retrain=True, **kwargs):
        # Check if optimization is needed
        if len(X) <= self.opt_tr_size:
            # Optimize the hyperparameters
            return super().fit(
                X,
                Y,
                model,
                hp=hp,
                pdis=pdis,
                retrain=retrain,
                **kwargs,
            )
        # Use existing hyperparameters
        hp, theta, parameters = self.get_hyperparams(hp, model)
        # Do not optimize hyperparameters
        sol = {
            "fun": inf,
            "x": theta,
            "hp": hp,
            "success": False,
            "nfev": 0,
            "nit": 0,
            "message": "No function values calculated.",
        }
        sol = OptimizeResult(**sol)
        # Get the full set of hyperparameters in the model
        sol = self.get_full_hp(sol, model)
        return sol

    def update_arguments(
        self,
        func=None,
        optimizer=None,
        bounds=None,
        use_update_pdis=None,
        get_prior_mean=None,
        use_stored_sols=None,
        round_hp=None,
        opt_tr_size=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            func: ObjectiveFunction class
                A class with the objective function used
                to optimize the hyperparameters.
            optimizer: Optimizer class
                A class with the used optimization method.
            bounds: HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                Most of the global optimizers are using boundary conditions.
                The bounds in this class will be used
                for the optimizer and func.
            use_update_pdis: bool
                Whether to update the prior distributions of
                the hyperparameters with the given boundary conditions.
            get_prior_mean: bool
                Whether to get the parameters of the prior mean
                in the solution.
            use_stored_sols: bool
                Whether to store the solutions.
            round_hp: int (optional)
                The number of decimals to round the hyperparameters to.
                If None, the hyperparameters are not rounded.
            opt_tr_size: int
                The maximum size of the training set before
                the hyperparameters are not optimized.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        super().update_arguments(
            func=func,
            optimizer=optimizer,
            bounds=bounds,
            use_update_pdis=use_update_pdis,
            get_prior_mean=get_prior_mean,
            use_stored_sols=use_stored_sols,
            round_hp=round_hp,
            dtype=dtype,
        )
        if opt_tr_size is not None:
            self.opt_tr_size = opt_tr_size
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            func=self.func,
            optimizer=self.optimizer,
            bounds=self.bounds,
            use_update_pdis=self.use_update_pdis,
            get_prior_mean=self.get_prior_mean,
            use_stored_sols=self.use_stored_sols,
            round_hp=self.round_hp,
            opt_tr_size=self.opt_tr_size,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict(sols=self.get_sols())
        return arg_kwargs, constant_kwargs, object_kwargs
