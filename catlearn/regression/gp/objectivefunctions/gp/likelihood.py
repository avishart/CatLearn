import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction


class LogLikelihood(ObjectiveFuction):
    def __init__(self, get_prior_mean=False, **kwargs):
        """
        The log-likelihood objective function that is used to
        optimize the hyperparameters.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
        """
        super().__init__(get_prior_mean=get_prior_mean, **kwargs)

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
        nlp = (
            0.5 * np.matmul(Y_p.T, coef).item(0) / prefactor2
            + 0.5 * n_data * np.log(prefactor2)
            + np.sum(np.log(np.diagonal(L)))
            + 0.5 * n_data * np.log(2.0 * np.pi)
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
        nlp_deriv = np.array([])
        KXX_inv = cho_solve((L, low), np.identity(n_data), check_finite=False)
        for para in parameters_set:
            if para == "prefactor":
                nlp_d = -np.matmul(Y_p.T, coef).item(0) / prefactor2 + n_data
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
