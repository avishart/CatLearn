import numpy as np
from .factorized_likelihood import FactorizedLogLikelihood


class FactorizedGPP(FactorizedLogLikelihood):
    def __init__(
        self,
        get_prior_mean=False,
        modification=False,
        ngrid=80,
        bounds=None,
        noise_optimizer=None,
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
            noise_optimizer : Noise line search optimizer class
                A line search optimization method for
                the relative-noise hyperparameter.
        """
        super().__init__(
            get_prior_mean=get_prior_mean,
            modification=modification,
            ngrid=ngrid,
            bounds=bounds,
            noise_optimizer=noise_optimizer,
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
        gpp_deriv = np.array([])
        D_n = D + np.exp(2 * noise)
        UDn = U / D_n
        KXX_inv = np.matmul(UDn, U.T)
        K_inv_diag = np.diag(KXX_inv)
        prefactor2 = np.mean(
            (np.matmul(UDn, UTY).reshape(-1) ** 2) / K_inv_diag
        )
        hp["prefactor"] = np.array([0.5 * np.log(prefactor2)])
        hp["noise"] = np.array([noise])
        coef_re = np.matmul(KXX_inv, Y_p).reshape(-1)
        co_Kinv = coef_re / K_inv_diag
        for para in parameters_set:
            if para == "prefactor":
                gpp_d = np.zeros((len(hp[para])))
            else:
                K_deriv = self.get_K_deriv(model, para, X=X, KXX=KXX)
                r_j, s_j = self.get_r_s_derivatives(K_deriv, KXX_inv, coef_re)
                gpp_d = (
                    np.mean(co_Kinv * (2.0 * r_j + co_Kinv * s_j), axis=-1)
                    / prefactor2
                ) + np.mean(s_j / K_inv_diag, axis=-1)
            gpp_deriv = np.append(gpp_deriv, gpp_d)
        gpp_deriv = gpp_deriv - self.logpriors(hp, pdis, jac=True) / n_data
        return gpp_deriv

    def get_r_s_derivatives(self, K_deriv, KXX_inv, coef):
        """
        Get the r and s vector that are products of the inverse and
        derivative covariance matrix
        """
        r_j = np.einsum("ji,di->dj", KXX_inv, np.matmul(K_deriv, -coef))
        s_j = np.einsum("ji,dji->di", KXX_inv, np.matmul(K_deriv, KXX_inv))
        return r_j, s_j

    def get_eig_fun(self, noise, hp, pdis, U, UTY, D, n_data, **kwargs):
        "Calculate GPP from Eigendecomposition for a noise value."
        D_n = D + np.exp(2.0 * noise)
        UDn = U / D_n
        K_inv_diag = np.einsum("ij,ji->i", UDn, U.T)
        prefactor = 0.5 * np.log(
            np.mean((np.matmul(UDn, UTY).reshape(-1) ** 2) / K_inv_diag)
        )
        gpp_v = (
            1
            - np.mean(np.log(K_inv_diag))
            + 2.0 * prefactor
            + np.log(2.0 * np.pi)
        )
        if pdis is not None:
            hp["prefactor"] = np.array([prefactor])
            hp["noise"] = np.array([noise]).reshape(-1)
        return gpp_v - self.logpriors(hp, pdis, jac=False) / n_data

    def get_all_eig_fun(self, noises, hp, pdis, U, UTY, D, n_data, **kwargs):
        """
        Calculate GPP from Eigendecompositions for all noise values
        from the list.
        """
        D_n = D + np.exp(2.0 * noises)
        UDn = U / D_n[:, None, :]
        K_inv_diag = np.einsum("dij,ji->di", UDn, U.T, optimize=True)
        prefactor = 0.5 * np.log(
            np.mean(
                (np.matmul(UDn, UTY).reshape((len(noises), n_data)) ** 2)
                / K_inv_diag,
                axis=1,
            )
        )
        gpp_v = (
            1.0
            - np.mean(np.log(K_inv_diag), axis=1)
            + 2.0 * prefactor
            + np.log(2.0 * np.pi)
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
            D_n = D + np.exp(2.0 * noise)
            UDn = U / D_n
            K_inv_diag = np.einsum("ij,ji->i", UDn, U.T)
            prefactor2 = np.mean(
                (np.matmul(UDn, UTY).reshape(-1) ** 2) / K_inv_diag
            )
            if self.modification:
                prefactor2 = (
                    (n_data / (n_data - len(theta))) * prefactor2
                    if n_data - len(theta) > 0
                    else prefactor2
                )
            hp["prefactor"] = np.array([0.5 * np.log(prefactor2)])
            hp["noise"] = np.array([noise])
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
