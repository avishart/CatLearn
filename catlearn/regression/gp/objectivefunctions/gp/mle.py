import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction


class MaximumLogLikelihood(ObjectiveFuction):
    def __init__(self, get_prior_mean=False, modification=False, **kwargs):
        """
        The Maximum log-likelihood objective function as
        a function of the hyperparameters.
        The prefactor hyperparameter is calculated from
        an analytical expression.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
            modification: bool
                Whether to modify the analytical prefactor value in the end.
                The prefactor hyperparameter becomes larger
                if modification=True.
        """
        # Set descriptor of the objective function
        self.use_analytic_prefactor = True
        self.use_optimized_noise = False
        # Set the arguments
        self.update_arguments(
            get_prior_mean=get_prior_mean,
            modification=modification,
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
        prefactor2 = np.matmul(Y_p.T, coef).item(0) / n_data
        prefactor = 0.5 * np.log(prefactor2)
        hp["prefactor"] = np.array([prefactor])
        nlp = (
            0.5 * n_data * (1 + np.log(2.0 * np.pi))
            + n_data * prefactor
            + np.sum(np.log(np.diagonal(L)))
        )
        nlp = nlp - self.logpriors(hp, pdis, jac=False)
        if jac:
            deriv = self.derivative(
                hp,
                parameters_set,
                model,
                X,
                KXX,
                L,
                low,
                coef,
                prefactor2,
                n_data,
                pdis,
                **kwargs,
            )
            self.update_solution(
                nlp,
                theta,
                hp,
                model,
                jac=jac,
                deriv=deriv,
                prefactor2=prefactor2,
                n_data=n_data,
            )
            return nlp, deriv
        self.update_solution(
            nlp,
            theta,
            hp,
            model,
            jac=jac,
            prefactor2=prefactor2,
            n_data=n_data,
        )
        return nlp

    def derivative(
        self,
        hp,
        parameters_set,
        model,
        X,
        KXX,
        L,
        low,
        coef,
        prefactor2,
        n_data,
        pdis,
        **kwargs,
    ):
        nlp_deriv = np.array([])
        KXX_inv = cho_solve((L, low), np.identity(n_data), check_finite=False)
        for para in parameters_set:
            if para == "prefactor":
                nlp_d = np.zeros((len(hp[para])))
            else:
                K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
                K_deriv_cho = self.get_K_inv_deriv(K_deriv, KXX_inv)
                nlp_d = (
                    (-0.5 / prefactor2)
                    * np.matmul(coef.T, np.matmul(K_deriv, coef)).reshape(-1)
                ) + (0.5 * K_deriv_cho)
            nlp_deriv = np.append(nlp_deriv, nlp_d)
        nlp_deriv = nlp_deriv - self.logpriors(hp, pdis, jac=True)
        return nlp_deriv

    def update_arguments(
        self,
        get_prior_mean=None,
        modification=None,
        **kwargs,
    ):
        """
        Update the objective function with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
            modification: bool
                Whether to modify the analytical prefactor value in the end.
                The prefactor hyperparameter becomes larger
                if modification=True.

        Returns:
            self: The updated object itself.
        """
        if get_prior_mean is not None:
            self.get_prior_mean = get_prior_mean
        if modification is not None:
            self.modification = modification
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
        prefactor2=None,
        n_data=None,
        **kwargs,
    ):
        """
        Update the solution of the optimization in terms of
        hyperparameters and model.
        The lowest objective function value is stored togeher
        with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        The prefactor hyperparameter are stored as a different value
        than the input since it is optimized analytically.
        """
        if fun < self.sol["fun"]:
            if self.modification:
                if n_data - len(theta) > 0:
                    prefactor2 = (n_data / (n_data - len(theta))) * prefactor2
                hp["prefactor"] = np.array([0.5 * np.log(prefactor2)])
            self.sol["x"] = np.concatenate(
                [hp[para] for para in sorted(hp.keys())]
            )
            self.sol["hp"] = hp.copy()
            self.sol["fun"] = fun
            if jac:
                self.sol["jac"] = deriv.copy()
            if self.get_prior_mean:
                self.sol["prior"] = self.get_prior_parameters(model)
        return self.sol

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            get_prior_mean=self.get_prior_mean,
            modification=self.modification,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
