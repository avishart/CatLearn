from numpy import asarray, diag, dot, empty, exp, full
from .model import (
    ModelProcess,
    Prior_mean,
    SE,
    HyperparameterFitter,
)
from ..objectivefunctions.tp.likelihood import LogLikelihood


class TProcess(ModelProcess):
    """
    The Student's T Process Regressor.
    The Student's T process uses Cholesky decomposition for
    inverting the kernel matrix.
    The hyperparameters can be optimized.
    """

    def __init__(
        self,
        prior=Prior_mean(),
        kernel=SE(use_derivatives=False, use_fingerprint=False),
        hpfitter=HyperparameterFitter(func=LogLikelihood()),
        hp={},
        use_derivatives=False,
        use_correction=True,
        a=1e-20,
        b=1e-20,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the Student's T Process Regressor.

        Parameters:
            prior: Prior class
                The prior given for new data.
            kernel: Kernel class
                The kernel function used for the kernel matrix.
            hpfitter: HyperparameterFitter class
                A class to optimize hyperparameters
            hp: dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives: bool
                Use derivatives/gradients for training and predictions.
            use_correction: bool
                Use the noise correction on the covariance matrix.
            a: float
                Hyperprior shape parameter for the inverse-gamma distribution
                of the prefactor.
            b: float
                Hyperprior scale parameter for the inverse-gamma distribution
                of the prefactor.
            dtype: type
                The data type of the arrays.
        """
        # Set default descriptors
        self.trained_model = False
        self.corr = 0.0
        self.features = []
        self.L = empty(0, dtype=dtype)
        self.low = False
        self.coef = empty(0, dtype=dtype)
        self.prefactor = 1.0
        # Set default hyperparameters
        self.hp = {"noise": asarray([-8.0], dtype=dtype)}
        # Set all the arguments
        self.update_arguments(
            prior=prior,
            kernel=kernel,
            hpfitter=hpfitter,
            hp=hp,
            use_derivatives=use_derivatives,
            use_correction=use_correction,
            a=a,
            b=b,
            dtype=dtype,
            **kwargs,
        )

    def update_arguments(
        self,
        prior=None,
        kernel=None,
        hpfitter=None,
        hp={},
        use_derivatives=None,
        use_correction=None,
        a=None,
        b=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the Model Process Regressor with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            prior: Prior class
                The prior given for new data.
            kernel: Kernel class
                The kernel function used for the kernel matrix.
            hpfitter: HyperparameterFitter class
                A class to optimize hyperparameters
            hp: dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives: bool
                Use derivatives/gradients for training and predictions.
            use_correction: bool
                Use the noise correction on the covariance matrix.
            a: float
                Hyperprior shape parameter for the inverse-gamma distribution
                of the prefactor.
            b: float
                Hyperprior scale parameter for the inverse-gamma distribution
                of the prefactor.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        super().update_arguments(
            prior=prior,
            kernel=kernel,
            hpfitter=hpfitter,
            hp=hp,
            use_derivatives=use_derivatives,
            use_correction=use_correction,
            dtype=dtype,
            **kwargs,
        )
        # The hyperprior shape parameter
        if a is not None:
            self.a = float(a)
        # The hyperprior scale parameter
        if b is not None:
            self.b = float(b)
        return self

    def get_gradients(self, features, hp, KXX, **kwargs):
        hp_deriv = {}
        n_data, m_data = len(features), len(KXX)
        if "noise" in hp:
            K_deriv = full(
                m_data,
                2.0 * exp(2.0 * self.hp["noise"][0]),
                dtype=self.dtype,
            )
            if "noise_deriv" in self.hp:
                K_deriv[n_data:] = 0.0
                hp_deriv["noise"] = asarray([diag(K_deriv)])
            else:
                hp_deriv["noise"] = asarray([diag(K_deriv)])
        if "noise_deriv" in hp:
            K_deriv = full(
                m_data,
                2.0 * exp(2.0 * self.hp["noise_deriv"][0]),
                dtype=self.dtype,
            )
            K_deriv[:n_data] = 0.0
            hp_deriv["noise_deriv"] = asarray([diag(K_deriv)])
        hp_deriv.update(self.kernel.get_gradients(features, hp, KXX=KXX))
        return hp_deriv

    def get_hyperprior_parameters(self, **kwargs):
        "Get the hyperprior parameters from the Student's T Process."
        return self.a, self.b

    def calculate_prefactor(self, features, targets, **kwargs):
        n2 = float(len(targets) - 2) if len(targets) > 1 else 0.0
        tcoef = dot(targets.reshape(-1), self.coef.reshape(-1))
        return (2.0 * self.b + tcoef) / (2.0 * self.a + n2)

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            prior=self.prior,
            kernel=self.kernel,
            hpfitter=self.hpfitter,
            hp=self.get_hyperparams(),
            use_derivatives=self.use_derivatives,
            use_correction=self.use_correction,
            a=self.a,
            b=self.b,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict(
            trained_model=self.trained_model,
            corr=self.corr,
            low=self.low,
            prefactor=self.prefactor,
        )
        # Get the objects made within the class
        object_kwargs = dict(features=self.features, L=self.L, coef=self.coef)
        return arg_kwargs, constant_kwargs, object_kwargs
