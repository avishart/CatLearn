from numpy import (
    append,
    empty,
    diagonal,
    log,
    matmul,
)
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
        a, b = self.get_hyperprior_parameters(model)
        coef, L, low, Y_p, KXX, n_data = self.coef_cholesky(model, X, Y)
        ycoef = 1.0 + (matmul(Y_p.T, coef).item(0) / (2.0 * b))
        nlp = log(diagonal(L)).sum() + 0.5 * (2.0 * a + n_data) * log(ycoef)
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
                a,
                b,
                ycoef,
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
        a,
        b,
        ycoef,
        n_data,
        pdis,
        **kwargs,
    ):
        nlp_deriv = empty(0, dtype=self.dtype)
        KXX_inv = self.get_cinv(L=L, low=low, n_data=n_data)
        for para in parameters_set:
            K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
            K_deriv_cho = self.get_K_inv_deriv(K_deriv, KXX_inv)
            nlp_d = (
                (-0.5 / ycoef)
                * ((a + 0.5 * n_data) / b)
                * matmul(coef.T, matmul(K_deriv, coef)).reshape(-1)
            ) + 0.5 * K_deriv_cho
            nlp_deriv = append(nlp_deriv, nlp_d)
        nlp_deriv = nlp_deriv - self.logpriors(hp, pdis, jac=True)
        return nlp_deriv

    def get_hyperprior_parameters(self, model, **kwargs):
        "Get the hyperprior parameters from the Student's T Process."
        a, b = model.get_hyperprior_parameters()
        return a, b
