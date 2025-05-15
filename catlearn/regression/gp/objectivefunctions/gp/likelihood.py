from numpy import append, diagonal, empty, log, matmul, pi
from ..objectivefunction import ObjectiveFuction


class LogLikelihood(ObjectiveFuction):
    def __init__(self, get_prior_mean=False, dtype=float, **kwargs):
        """
        The log-likelihood objective function that is used to
        optimize the hyperparameters.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(get_prior_mean=get_prior_mean, dtype=dtype, **kwargs)

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
        prefactor2 = self.get_prefactor2(model)
        nlp = 0.5 * (
            matmul(Y_p.T, coef).item(0) / prefactor2
            + n_data * log(prefactor2)
            + 2.0 * log(diagonal(L)).sum()
            + n_data * log(2.0 * pi)
        )
        nlp = nlp - self.logpriors(hp, pdis, jac=False)
        if jac:
            return nlp, self.derivative(
                hp,
                parameters_set,
                model,
                X,
                Y_p,
                KXX,
                L,
                low,
                coef,
                prefactor2,
                n_data,
                pdis,
                **kwargs,
            )
        return nlp

    def derivative(
        self,
        hp,
        parameters_set,
        model,
        X,
        Y_p,
        KXX,
        L,
        low,
        coef,
        prefactor2,
        n_data,
        pdis,
        **kwargs,
    ):
        nlp_deriv = empty(0, dtype=self.dtype)
        KXX_inv = self.get_cinv(L=L, low=low, n_data=n_data)
        for para in parameters_set:
            if para == "prefactor":
                nlp_d = -matmul(Y_p.T, coef).item(0) / prefactor2 + n_data
            else:
                K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
                K_deriv_cho = self.get_K_inv_deriv(K_deriv, KXX_inv)
                nlp_d = (
                    (-0.5 / prefactor2)
                    * matmul(coef.T, matmul(K_deriv, coef)).reshape(-1)
                ) + (0.5 * K_deriv_cho)
            nlp_deriv = append(nlp_deriv, nlp_d)
        nlp_deriv = nlp_deriv - self.logpriors(hp, pdis, jac=True)
        return nlp_deriv
