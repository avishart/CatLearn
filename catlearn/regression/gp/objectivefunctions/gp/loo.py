import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction


class LOO(ObjectiveFuction):
    def __init__(
        self,
        get_prior_mean=False,
        use_analytic_prefactor=True,
        **kwargs,
    ):
        """
        The leave-one-out objective function that is used to
        optimize the hyperparameters.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
            use_analytic_prefactor: bool
                Whether to calculate the analytical prefactor value in the end.
        """
        # Set descriptor of the objective function
        self.use_optimized_noise = False
        # Set the arguments
        self.update_arguments(
            get_prior_mean=get_prior_mean,
            use_analytic_prefactor=use_analytic_prefactor,
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
        coef, L, low, Y_p, KXX, n_data = self.coef_cholesky(model, X, Y)
        KXX_inv, K_inv_diag, coef_re, co_Kinv = self.get_co_Kinv(
            L,
            low,
            n_data,
            coef,
        )
        loo_v = np.mean(co_Kinv**2)
        loo_v = loo_v - self.logpriors(hp, pdis, jac=False) / n_data
        if jac:
            deriv = self.derivative(
                hp,
                parameters_set,
                model,
                X,
                KXX,
                KXX_inv,
                K_inv_diag,
                coef_re,
                co_Kinv,
                n_data,
                pdis,
                **kwargs,
            )
            self.update_solution(
                loo_v,
                theta,
                hp,
                model,
                jac=jac,
                deriv=deriv,
                coef_re=coef_re,
                K_inv_diag=K_inv_diag,
                co_Kinv=co_Kinv,
            )
            return loo_v, deriv
        self.update_solution(
            loo_v,
            theta,
            hp,
            model,
            jac=jac,
            coef_re=coef_re,
            K_inv_diag=K_inv_diag,
            co_Kinv=co_Kinv,
        )
        return loo_v

    def derivative(
        self,
        hp,
        parameters_set,
        model,
        X,
        KXX,
        KXX_inv,
        K_inv_diag,
        coef_re,
        co_Kinv,
        n_data,
        pdis,
        **kwargs,
    ):
        loo_deriv = np.array([])
        for para in parameters_set:
            if para == "prefactor":
                loo_d = np.zeros((len(hp[para])))
            else:
                K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
                r_j, s_j = self.get_r_s_derivatives(K_deriv, KXX_inv, coef_re)
                loo_d = 2.0 * np.mean(
                    (co_Kinv / K_inv_diag) * (r_j + s_j * co_Kinv),
                    axis=-1,
                )
            loo_deriv = np.append(loo_deriv, loo_d)
        loo_deriv = loo_deriv - self.logpriors(hp, pdis, jac=True) / n_data
        return loo_deriv

    def update_arguments(
        self,
        get_prior_mean=None,
        use_analytic_prefactor=None,
        **kwargs,
    ):
        """
        Update the objective function with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
            use_analytic_prefactor: bool
                Whether to calculate the analytical prefactor value in the end.

        Returns:
            self: The updated object itself.
        """
        if get_prior_mean is not None:
            self.get_prior_mean = get_prior_mean
        if use_analytic_prefactor is not None:
            self.use_analytic_prefactor = use_analytic_prefactor
        # Always reset the solution when the objective function is changed
        self.reset_solution()
        return self

    def update_solution(
        self,
        fun,
        theta,
        hp,
        model,
        jac=False,
        deriv=None,
        coef_re=None,
        K_inv_diag=None,
        co_Kinv=None,
        **kwargs,
    ):
        """
        Update the solution of the optimization in terms of
        hyperparameters and model.
        The lowest objective function value is stored togeher
        with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        The prefactor hyperparameter are stored as a different value
        than the input since it is optimized analytically
        if use_analytic_prefactor=True.
        """
        if fun < self.sol["fun"]:
            if self.use_analytic_prefactor:
                prefactor2 = np.mean(co_Kinv * coef_re) - (
                    np.mean(coef_re / np.sqrt(K_inv_diag)) ** 2
                )
                hp["prefactor"] = np.array([0.5 * np.log(prefactor2)])
                self.sol["x"] = np.concatenate(
                    [hp[para] for para in sorted(hp.keys())]
                )
            else:
                self.sol["x"] = theta.copy()
            self.sol["hp"] = hp.copy()
            self.sol["fun"] = fun
            if jac:
                self.sol["jac"] = deriv.copy()
            if self.get_prior_mean:
                self.sol["prior"] = self.get_prior_parameters(model)
        return self.sol

    def get_co_Kinv(self, L, low, n_data, coef):
        "Get the inverse covariance matrix and diagonal products."
        KXX_inv = cho_solve((L, low), np.identity(n_data), check_finite=False)
        K_inv_diag = np.diag(KXX_inv)
        coef_re = coef.reshape(-1)
        co_Kinv = coef_re / K_inv_diag
        return KXX_inv, K_inv_diag, coef_re, co_Kinv

    def get_r_s_derivatives(self, K_deriv, KXX_inv, coef):
        """
        Get the r and s vector that are products of the inverse and
        derivative covariance matrix
        """
        r_j = np.einsum("ji,di->dj", KXX_inv, np.matmul(K_deriv, -coef))
        s_j = np.einsum("ji,dji->di", KXX_inv, np.matmul(K_deriv, KXX_inv))
        return r_j, s_j

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            get_prior_mean=self.get_prior_mean,
            use_analytic_prefactor=self.use_analytic_prefactor,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
