from numpy import append, empty
from .loo import LOO


class GPE(LOO):
    def __init__(self, get_prior_mean=False, dtype=float, **kwargs):
        """
        The Geissers predictive mean square error objective function as
        a function of the hyperparameters.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
             dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set descriptor of the objective function
        self.use_analytic_prefactor = False
        self.use_optimized_noise = False
        # Set the arguments
        self.update_arguments(
            get_prior_mean=get_prior_mean,
            dtype=dtype,
            **kwargs,
        )

    def function(
        self,
        theta,
        parameters,
        model,
        X,
        Y,
        pdis=None,
        jac=False,
        **kwargs,
    ):
        hp, parameters_set = self.make_hp(theta, parameters)
        model = self.update_model(model, hp)
        coef, L, low, _, KXX, n_data = self.coef_cholesky(model, X, Y)
        KXX_inv, K_inv_diag, coef_re, co_Kinv = self.get_co_Kinv(
            L,
            low,
            n_data,
            coef,
        )
        K_inv_diag_rev = 1.0 / K_inv_diag
        prefactor2 = self.get_prefactor2(model)
        gpe_v = (co_Kinv**2).mean() + prefactor2 * K_inv_diag_rev.mean()
        gpe_v = gpe_v - self.logpriors(hp, pdis, jac=False) / n_data
        if jac:
            return gpe_v, self.derivative(
                hp,
                parameters_set,
                model,
                X,
                KXX,
                KXX_inv,
                K_inv_diag_rev,
                coef_re,
                co_Kinv,
                prefactor2,
                n_data,
                pdis,
                **kwargs,
            )
        return gpe_v

    def derivative(
        self,
        hp,
        parameters_set,
        model,
        X,
        KXX,
        KXX_inv,
        K_inv_diag_rev,
        coef_re,
        co_Kinv,
        prefactor2,
        n_data,
        pdis,
        **kwargs,
    ):
        gpe_deriv = empty(0, dtype=self.dtype)
        for para in parameters_set:
            if para == "prefactor":
                gpe_d = 2.0 * prefactor2 * K_inv_diag_rev.mean()
            else:
                K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
                r_j, s_j = self.get_r_s_derivatives(K_deriv, KXX_inv, coef_re)
                gpe_d = 2.0 * (
                    (co_Kinv * K_inv_diag_rev) * (r_j + s_j * co_Kinv)
                ).mean(axis=-1) + prefactor2 * (
                    s_j * (K_inv_diag_rev * K_inv_diag_rev)
                ).mean(
                    axis=-1
                )
            gpe_deriv = append(gpe_deriv, gpe_d)
        gpe_deriv = gpe_deriv - self.logpriors(hp, pdis, jac=True) / n_data
        return gpe_deriv

    def update_arguments(self, get_prior_mean=None, dtype=None, **kwargs):
        """
        Update the objective function with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            get_prior_mean: bool
                Whether to get the parameters of the prior mean
                in the solution.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated object itself.
        """
        # Set the arguments of the parent class
        super().update_arguments(
            get_prior_mean=get_prior_mean,
            dtype=dtype,
        )
        return self

    def update_solution(
        self,
        fun,
        theta,
        hp,
        model,
        jac=False,
        deriv=None,
        **kwargs,
    ):
        """
        Update the solution of the optimization in terms of
        hyperparameters and model.
        The lowest objective function value is stored togehe
          with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        """
        if fun < self.sol["fun"]:
            self.sol["fun"] = fun
            self.sol["x"] = theta.copy()
            self.sol["hp"] = hp.copy()
            if jac:
                self.sol["jac"] = deriv.copy()
            if self.get_prior_mean:
                self.sol["prior"] = self.get_prior_parameters(model)
        return self.sol

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(get_prior_mean=self.get_prior_mean)
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
