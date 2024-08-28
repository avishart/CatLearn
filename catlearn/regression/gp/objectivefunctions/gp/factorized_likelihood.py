import numpy as np
from ..objectivefunction import ObjectiveFuction


class FactorizedLogLikelihood(ObjectiveFuction):
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
        The factorized log-likelihood objective function that is used
        to optimize the hyperparameters.
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
        # Set descriptor of the objective function
        self.use_analytic_prefactor = True
        self.use_optimized_noise = True
        # Set default bounds
        if bounds is None:
            from ...hpboundary.hptrans import VariableTransformation

            bounds = VariableTransformation(bounds=None)
        # Set default noise line optimizer
        if noise_optimizer is None:
            from ...optimizers.noisesearcher import NoiseFineGridSearch

            noise_optimizer = NoiseFineGridSearch(
                maxiter=1000,
                tol=1e-5,
                optimize=True,
                multiple_min=False,
                ngrid=ngrid,
                loops=2,
            )
        # Set the arguments
        self.update_arguments(
            get_prior_mean=get_prior_mean,
            modification=modification,
            ngrid=ngrid,
            bounds=bounds,
            noise_optimizer=noise_optimizer,
            **kwargs,
        )

    def update_arguments(
        self,
        get_prior_mean=None,
        modification=None,
        ngrid=None,
        bounds=None,
        noise_optimizer=None,
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
            ngrid: int
                Number of grid points that are searched in
                the relative-noise hyperparameter.
            bounds: Boundary_conditions class
                A class of the boundary conditions of
                the relative-noise hyperparameter.
            noise_optimizer : Noise line search optimizer class
                A line search optimization method for
                the relative-noise hyperparameter.

        Returns:
            self: The updated object itself.
        """
        if get_prior_mean is not None:
            self.get_prior_mean = get_prior_mean
        if modification is not None:
            self.modification = modification
        if ngrid is not None:
            self.ngrid = int(ngrid)
        if bounds is not None:
            self.bounds = bounds.copy()
        if noise_optimizer is not None:
            self.noise_optimizer = noise_optimizer.copy()
        # Always reset the solution when the objective function is changed
        self.reset_solution()
        return self

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
        noise, nlp = self.maximize_noise(
            parameters,
            model,
            X,
            Y,
            pdis,
            hp,
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
                **kwargs,
            )
            self.update_solution(
                nlp,
                theta,
                hp,
                model,
                jac=jac,
                deriv=deriv,
                noise=noise,
                UTY=UTY,
                D=D,
                n_data=n_data,
            )
            return nlp, deriv
        self.update_solution(
            nlp,
            theta,
            hp,
            model,
            jac=jac,
            noise=noise,
            UTY=UTY,
            D=D,
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
        D,
        U,
        Y_p,
        UTY,
        noise,
        pdis,
        **kwargs,
    ):
        nlp_deriv = np.array([])
        D_n = D + np.exp(2.0 * noise)
        prefactor2 = np.mean(UTY / D_n)
        hp["prefactor"] = np.array([0.5 * np.log(prefactor2)])
        hp["noise"] = np.array([noise])
        KXX_inv = np.matmul(U / D_n, U.T)
        coef = np.matmul(KXX_inv, Y_p)
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

    def get_eig_fun(self, noise, hp, pdis, UTY, D, n_data, **kwargs):
        "Calculate log-likelihood from Eigendecomposition for a noise value."
        D_n = D + np.exp(2.0 * noise)
        prefactor = 0.5 * np.log(np.mean(UTY / D_n))
        nlp = (
            0.5 * n_data * (1 + np.log(2.0 * np.pi))
            + (n_data * prefactor)
            + 0.5 * np.sum(np.log(D_n))
        )
        if pdis is not None:
            hp["prefactor"] = np.array([prefactor])
            hp["noise"] = np.array([noise]).reshape(-1)
        return nlp - self.logpriors(hp, pdis, jac=False)

    def get_all_eig_fun(self, noises, hp, pdis, UTY, D, n_data, **kwargs):
        """
        Calculate log-likelihood from Eigendecompositions for
        all noise values from the list.
        """
        D_n = D + np.exp(2.0 * noises)
        prefactor = 0.5 * np.log(np.mean(UTY / D_n, axis=1))
        nlp = (0.5 * n_data * (1 + np.log(2.0 * np.pi))) + (
            (n_data * prefactor) + (0.5 * np.sum(np.log(D_n), axis=1))
        )
        if pdis is not None:
            hp["prefactor"] = prefactor.reshape(-1, 1)
            hp["noise"] = noises
        return nlp - self.logpriors(hp, pdis, jac=False)

    def make_noise_list(self, model, X, Y, **kwargs):
        "Make the list of noises."
        return self.bounds.make_single_line(
            parameter="noise", ngrid=self.ngrid
        ).reshape(-1, 1)

    def maximize_noise(
        self,
        parameters,
        model,
        X,
        Y,
        pdis,
        hp,
        UTY,
        D,
        n_data,
        **kwargs,
    ):
        "Find the maximum relative-noise with a grid method."
        noises = self.make_noise_list(model, X, Y)
        # Make the function arguments
        func_args = (hp.copy(), pdis, UTY, D, n_data)
        # Calculate function values for line coordinates
        sol = self.noise_optimizer.run(
            self, noises, ["noise"], model, X, Y, pdis, func_args=func_args
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
            prefactor2 = np.mean(UTY / D_n)
            if self.modification:
                if n_data - len(theta) > 0:
                    prefactor2 = (n_data / (n_data - len(theta))) * prefactor2
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

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            get_prior_mean=self.get_prior_mean,
            modification=self.modification,
            ngrid=self.ngrid,
            bounds=self.bounds,
            noise_optimizer=self.noise_optimizer,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
