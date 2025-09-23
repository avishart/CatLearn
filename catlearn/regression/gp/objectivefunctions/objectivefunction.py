from numpy import (
    array,
    asarray,
    append,
    diag,
    einsum,
    empty,
    finfo,
    identity,
    inf,
    log,
    matmul,
    where,
    zeros,
)
from numpy.linalg import eigh, LinAlgError
from scipy.linalg import cho_factor, cho_solve, eigh as scipy_eigh
import warnings


class ObjectiveFuction:
    """
    The objective function that is used to optimize the hyperparameters.
    """

    def __init__(self, get_prior_mean=False, dtype=float, **kwargs):
        """
        Initialize the objective function.

        Parameters:
            get_prior_mean: bool
                Whether to get the parameters of the prior mean
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
        """
        The function call that calculate the objective function.

        Parameters:
            theta: (H) array of floats
                An array with the hyperparameter values used for
                the objective function.
            parameters: (H) list of strings
                A list of names of the hyperparameters.
            model: Model
                The Machine Learning Model with kernel and prior that
                are optimized.
            X: (N,D) array
                Training features with N data points and D dimensions.
            Y: (N,1) array or (N,D+1) array
                Training targets without or with derivatives with
                N data points.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            jac: bool
                Whether to get the derivatives of the objective function
                wrt. the hyperparameters.

        Returns:
            float: The objective function value.
            and/or
            (H) array: The derivative of the objective function value
                wrt. the hyperparameters if jac=True.
        """
        raise NotImplementedError()

    def derivative(self, **kwargs):
        """
        The derivative of the objective function wrt. the hyperparameters.
        """
        raise NotImplementedError()

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the data type
        self.dtype = dtype
        return self

    def set_seed(self, seed, **kwargs):
        """
        Set the random seed.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        return self

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
        # Set the data type
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype=dtype)
        # Set the get_prior_mean
        if get_prior_mean is not None:
            self.get_prior_mean = get_prior_mean
        # Always reset the solution when the objective function is changed
        self.reset_solution()
        return self

    def reset_solution(self):
        """
        Reset the solution of the optimization in terms of
        the hyperparameters and model.
        """
        self.sol = {"fun": inf, "x": empty(0, dtype=self.dtype), "hp": {}}
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
        The lowest objective function value is stored togeher
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

    def get_solution(self, sol, parameters, model, X, Y, pdis=None, **kwargs):
        """
        Get the solution of the optimization in terms of
        hyperparameters and model.
        """
        if self.sol["fun"] > sol["fun"]:
            sol["hp"] = self.make_hp(sol["x"], parameters)[0]
            if self.get_prior_mean:
                sol["prior"] = self.get_prior_parameters(model, X=X, Y=Y)
            return sol
        sol["fun"] = self.sol["fun"]
        sol["x"] = self.sol["x"].copy()
        sol["hp"] = self.sol["hp"].copy()
        if "jac" in self.sol.keys():
            sol["jac"] = self.sol["jac"].copy()
        if "prior" in self.sol.keys():
            sol["prior"] = self.sol["prior"].copy()
        return sol

    def get_stored_solution(self, **kwargs):
        """
        Get the stored solution of the optimization of the hyperparameters
        within only the objective function.
        """
        return self.sol

    def make_hp(self, theta, parameters, **kwargs):
        "Make hyperparameter dictionary from lists"
        theta = asarray(theta)
        parameters_set = sorted(set(parameters))
        parameters = asarray(parameters)
        hp = {
            para_s: self.numeric_limits(theta[parameters == para_s])
            for para_s in parameters_set
        }
        return hp, parameters_set

    def get_hyperparams(self, model, **kwargs):
        "Get the hyperparameters for the model and the kernel."
        return model.get_hyperparams()

    def numeric_limits(self, a, dh=None):
        """
        Replace hyperparameters if they are outside of
        the numeric limits in log-space.
        """
        if dh is None:
            dh = 0.1 * log(finfo(self.dtype).max)
        return where(-dh < a, where(a < dh, a, dh), -dh)

    def update_model(self, model, hp, **kwargs):
        "Update the the machine learning model with the hyperparameters."
        model.set_hyperparams(hp)
        return model

    def kxx_reg(self, model, X, **kwargs):
        "Get covariance matrix with regularization."
        KXX = model.get_kernel(X, get_derivatives=model.use_derivatives)
        KXX_n = model.add_regularization(KXX, len(X), overwrite=False)
        return KXX_n, KXX, len(KXX)

    def kxx_corr(self, model, X, **kwargs):
        "Get covariance matrix with or without noise correction."
        # Calculate the kernel with and without noise
        KXX = model.get_kernel(X, get_derivatives=model.use_derivatives)
        n_data = len(KXX)
        KXX = self.add_correction(model, KXX, n_data)
        return KXX, n_data

    def add_correction(self, model, KXX, n_data, **kwargs):
        "Add noise correction to covariance matrix."
        corr = model.get_correction(diag(KXX))
        if corr > 0.0:
            KXX[range(n_data), range(n_data)] += corr
        return KXX

    def y_prior(self, X, Y, model, L=None, low=None, **kwargs):
        "Update prior and subtract to target."
        Y_p = array(Y, dtype=self.dtype)
        model.update_priormean(X, Y_p, L=L, low=low, **kwargs)
        get_derivatives = model.get_use_derivatives()
        pmean = model.get_priormean(
            X,
            Y_p,
            get_derivatives=get_derivatives,
        )
        Y_p -= pmean
        if get_derivatives:
            return Y_p.T.reshape(-1, 1)
        return Y_p[:, 0:1]

    def coef_cholesky(self, model, X, Y, **kwargs):
        "Calculate the coefficients by using Cholesky decomposition."
        # Calculate the kernel with and without noise
        KXX_n, KXX, n_data = self.kxx_reg(model, X)
        # Cholesky decomposition
        L, low = cho_factor(KXX_n)
        # Subtract the prior mean to the training target
        Y_p = self.y_prior(X, Y, model, L=L, low=low)
        # Get the coefficients
        coef = cho_solve((L, low), Y_p, check_finite=False)
        return coef, L, low, Y_p, KXX, n_data

    def get_eig(self, model, X, Y, **kwargs):
        "Calculate the eigenvalues."
        # Calculate the kernel with and without noise
        KXX, n_data = self.kxx_corr(model, X)
        # Eigendecomposition
        try:
            D, U = eigh(KXX)
        except LinAlgError:
            warnings.warn(
                "Eigendecomposition failed, using scipy.eigh instead."
            )
            # More robust but slower eigendecomposition
            D, U = scipy_eigh(KXX, driver="ev")
        # Subtract the prior mean to the training target
        Y_p = self.y_prior(X, Y, model, D=D, U=U)
        UTY = matmul(U.T, Y_p).reshape(-1) ** 2
        return D, U, Y_p, UTY, KXX, n_data

    def get_cinv_model(self, model, X, Y, check_finite=False, **kwargs):
        "Get the inverse covariance matrix from the model."
        coef, L, low, Y_p, KXX, n_data = self.coef_cholesky(model, X, Y)
        cinv = self.get_cinv(
            L,
            low,
            n_data,
            check_finite=check_finite,
            **kwargs,
        )
        return coef, cinv, Y_p, KXX, n_data

    def get_cinv(self, L, low, n_data, check_finite=False, **kwargs):
        "Get the inverse covariance matrix."
        return cho_solve(
            (L, low),
            identity(
                n_data,
                dtype=self.dtype,
            ),
            check_finite=check_finite,
        )

    def logpriors(self, hp, pdis=None, jac=False, **kwargs):
        "Log of the prior distribution value for the hyperparameters."
        # If no prior distribution is used for the hyperparameters
        if pdis is None:
            return 0.0
        # If the log probability is calculated
        if not jac:
            lprior = 0.0
            for para, value in hp.items():
                if para in pdis.keys():
                    lprior = lprior + pdis[para].ln_pdf(value)
            if isinstance(lprior, float):
                return lprior
            return lprior.reshape(-1)
        # Derivate of the log probability wrt. the hyperparameters
        lprior_deriv = empty(0, dtype=self.dtype)
        for para, value in hp.items():
            if para in pdis.keys():
                lprior_deriv = append(
                    lprior_deriv,
                    asarray(
                        pdis[para].ln_deriv(value),
                        dtype=self.dtype,
                    ).reshape(-1),
                )
            else:
                lprior_deriv = append(
                    lprior_deriv,
                    zeros((len(value)), dtype=self.dtype),
                )
        return lprior_deriv

    def get_K_inv_deriv(self, K_deriv, KXX_inv, **kwargs):
        """
        Get the diagonal elements of the matrix product of
        the inverse and derivative covariance matrix.
        """
        return einsum("ij,dji->d", KXX_inv, K_deriv)

    def get_K_deriv(self, model, parameter, X, KXX, **kwargs):
        "Get the gradient of the covariance matrix wrt. the hyperparameter."
        K_deriv = model.get_gradients(X, [parameter], KXX=KXX)[parameter]
        return K_deriv

    def get_prefactor2(self, model, **kwargs):
        "Get the squared prefactor hyperparameter in linear (exp) space."
        return model.get_prefactor()

    def get_prior_parameters(self, model, **kwargs):
        "Get the prior parameters."
        return model.get_prior_parameters(**kwargs)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(get_prior_mean=self.get_prior_mean, dtype=self.dtype)
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)
