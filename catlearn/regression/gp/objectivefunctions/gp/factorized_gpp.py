from numpy import (
    append,
    asarray,
    concatenate,
    diag,
    einsum,
    empty,
    exp,
    log,
    matmul,
    pi,
    zeros,
)
from .factorized_likelihood import (
    FactorizedLogLikelihood,
    VariableTransformation,
)


class FactorizedGPP(FactorizedLogLikelihood):
    def __init__(
        self,
        get_prior_mean=False,
        modification=False,
        ngrid=80,
        bounds=VariableTransformation(),
        noise_optimizer=None,
        dtype=float,
        **kwargs,
    ):
        """
        The factorized Geissers surrogate predictive probability
        objective function that is used to optimize the hyperparameters.
        The prefactor hyperparameter is determined from
        an analytical expression.
        An eigendecomposition is performed to get the eigenvalues.
        The relative-noise hyperparameter can be searched from
        a single eigendecomposition for each length-scale hyperparameter.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
            modification: bool
                Whether to modify the analytical prefactor value in the end.
                The prefactor hyperparameter becomes larger
                if modification=True.
            ngrid: int
                Number of grid points that are searched in
                the relative-noise hyperparameter.
            bounds: Boundary_conditions class
                A class of the boundary conditions of
                the relative-noise hyperparameter.
            noise_optimizer: Noise line search optimizer class
                A line search optimization method for
                the relative-noise hyperparameter.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        super().__init__(
            get_prior_mean=get_prior_mean,
            modification=modification,
            ngrid=ngrid,
            bounds=bounds,
            noise_optimizer=noise_optimizer,
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
        D, U, Y_p, UTY, KXX, n_data = self.get_eig(model, X, Y)
        noise, gpp_v = self.maximize_noise(
            parameters,
            model,
            X,
            Y,
            pdis,
            hp,
            U,
            UTY,
            D,
            n_data,
        )
        if jac:
            deriv = self.derivative(
                hp,
                parameters_set,
                model,
                X,
                KXX,
                D,
                U,
                Y_p,
                UTY,
                noise,
                pdis,
                n_data,
                **kwargs,
            )
            self.update_solution(
                gpp_v,
                theta,
                hp,
                model,
                jac=jac,
                deriv=deriv,
                noise=noise,
                UTY=UTY,
                U=U,
                D=D,
                n_data=n_data,
            )
            return gpp_v, deriv
        self.update_solution(
            gpp_v,
            theta,
            hp,
            model,
            jac=jac,
            noise=noise,
            UTY=UTY,
            U=U,
            D=D,
            n_data=n_data,
        )
        return gpp_v

    def derivative(
        self,
        hp,
        parameters_set,
        model,
        X,
        KXX,
        D,
        U,
        Y_p,
        UTY,
        noise,
        pdis,
        n_data,
        **kwargs,
    ):
        gpp_deriv = empty(0, dtype=self.dtype)
        D_n = D + exp(2.0 * noise)
        UDn = U / D_n
        KXX_inv = matmul(UDn, U.T)
        K_inv_diag = diag(KXX_inv)
        prefactor2 = ((matmul(UDn, UTY).reshape(-1) ** 2) / K_inv_diag).mean()
        hp["prefactor"] = asarray([0.5 * log(prefactor2)])
        hp["noise"] = asarray([noise])
        coef_re = matmul(KXX_inv, Y_p).reshape(-1)
        co_Kinv = coef_re / K_inv_diag
        for para in parameters_set:
            if para == "prefactor":
                gpp_d = zeros((len(hp[para])), dtype=self.dtype)
            else:
                K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
                r_j, s_j = self.get_r_s_derivatives(K_deriv, KXX_inv, coef_re)
                gpp_d = (
                    (co_Kinv * (2.0 * r_j + co_Kinv * s_j)).mean(axis=-1)
                    / prefactor2
                ) + (s_j / K_inv_diag).mean(axis=-1)
            gpp_deriv = append(gpp_deriv, gpp_d)
        gpp_deriv = gpp_deriv - self.logpriors(hp, pdis, jac=True) / n_data
        return gpp_deriv

    def get_r_s_derivatives(self, K_deriv, KXX_inv, coef):
        """
        Get the r and s vector that are products of the inverse and
        derivative covariance matrix
        """
        r_j = einsum("ji,di->dj", KXX_inv, matmul(K_deriv, -coef))
        s_j = einsum("ji,dji->di", KXX_inv, matmul(K_deriv, KXX_inv))
        return r_j, s_j

    def get_eig_fun(self, noise, hp, pdis, U, UTY, D, n_data, **kwargs):
        "Calculate GPP from Eigendecomposition for a noise value."
        D_n = D + exp(2.0 * noise)
        UDn = U / D_n
        K_inv_diag = einsum("ij,ji->i", UDn, U.T)
        prefactor = 0.5 * log(
            ((matmul(UDn, UTY).reshape(-1) ** 2) / K_inv_diag).mean()
        )
        gpp_v = 1.0 - log(K_inv_diag).mean() + 2.0 * prefactor + log(2.0 * pi)
        if pdis is not None:
            hp["prefactor"] = asarray([prefactor])
            hp["noise"] = asarray([noise]).reshape(-1)
        return gpp_v - self.logpriors(hp, pdis, jac=False) / n_data

    def get_all_eig_fun(self, noises, hp, pdis, U, UTY, D, n_data, **kwargs):
        """
        Calculate GPP from Eigendecompositions for all noise values
        from the list.
        """
        D_n = D + exp(2.0 * noises)
        UDn = U / D_n[:, None, :]
        K_inv_diag = einsum("dij,ji->di", UDn, U.T, optimize=True)
        prefactor = 0.5 * log(
            (
                (matmul(UDn, UTY).reshape((len(noises), n_data)) ** 2)
                / K_inv_diag
            ).mean(axis=1)
        )
        gpp_v = (
            1.0
            - log(K_inv_diag).mean(axis=1)
            + 2.0 * prefactor
            + log(2.0 * pi)
        )
        if pdis is not None:
            hp["prefactor"] = prefactor.reshape(-1, 1)
            hp["noise"] = noises
        return gpp_v - self.logpriors(hp, pdis, jac=False) / n_data

    def maximize_noise(
        self,
        parameters,
        model,
        X,
        Y,
        pdis,
        hp,
        U,
        UTY,
        D,
        n_data,
        **kwargs,
    ):
        "Find the maximum relative-noise with a grid method."
        noises = self.make_noise_list(model, X, Y)
        # Make the function arguments
        func_args = (hp.copy(), pdis, U, UTY, D, n_data)
        # Calculate function values for line coordinates
        sol = self.noise_optimizer.run(
            self,
            noises,
            ["noise"],
            model,
            X,
            Y,
            pdis,
            func_args=func_args,
        )
        # Find the minimum value
        return sol["x"][0], sol["fun"]

    def update_solution(
        self,
        fun,
        theta,
        hp,
        model,
        jac=False,
        deriv=None,
        noise=None,
        UTY=None,
        U=None,
        D=None,
        n_data=None,
        **kwargs,
    ):
        """
        Update the solution of the optimization in terms of
        hyperparameters and model.
        The lowest objective function value is stored togeher
        with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        The prefactor and relative-noise hyperparameters are
        stored as different values
        than the input since they are optimized analytically
        and numerically, respectively.
        """
        if fun < self.sol["fun"]:
            D_n = D + exp(2.0 * noise)
            UDn = U / D_n
            K_inv_diag = einsum("ij,ji->i", UDn, U.T)
            prefactor2 = (
                (matmul(UDn, UTY).reshape(-1) ** 2) / K_inv_diag
            ).mean()
            if self.modification:
                prefactor2 = (
                    (n_data / (n_data - len(theta))) * prefactor2
                    if n_data - len(theta) > 0
                    else prefactor2
                )
            hp["prefactor"] = asarray([0.5 * log(prefactor2)])
            hp["noise"] = asarray([noise])
            self.sol["x"] = concatenate(
                [hp[para] for para in sorted(hp.keys())]
            )
            self.sol["hp"] = hp.copy()
            self.sol["fun"] = fun
            if jac:
                self.sol["jac"] = deriv.copy()
            if self.get_prior_mean:
                self.sol["prior"] = self.get_prior_parameters(model)
        return self.sol
