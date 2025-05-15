from numpy import asarray, array, diag, empty, exp, full
from .model import (
    ModelProcess,
    Prior_mean,
    SE,
    HyperparameterFitter,
    LogLikelihood,
)


class GaussianProcess(ModelProcess):
    def __init__(
        self,
        prior=Prior_mean(),
        kernel=SE(use_derivatives=False, use_fingerprint=False),
        hpfitter=HyperparameterFitter(func=LogLikelihood()),
        hp={},
        use_derivatives=False,
        use_correction=True,
        dtype=float,
        **kwargs
    ):
        """
        The Gaussian Process Regressor.
        The Gaussian process uses Cholesky decomposition for
        inverting the kernel matrix.
        The hyperparameters can be optimized.

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
        self.hp = {
            "noise": asarray([-8.0], dtype=dtype),
            "prefactor": asarray([0.0], dtype=dtype),
        }
        # Set all the arguments
        self.update_arguments(
            prior=prior,
            kernel=kernel,
            hpfitter=hpfitter,
            hp=hp,
            use_derivatives=use_derivatives,
            use_correction=use_correction,
            dtype=dtype,
            **kwargs
        )

    def set_hyperparams(self, new_params, **kwargs):
        # Set the hyperparameters in the parent class
        super().set_hyperparams(new_params, **kwargs)
        # Set the prefactor hyperparameter
        if "prefactor" in new_params:
            self.hp["prefactor"] = array(
                new_params["prefactor"],
                dtype=self.dtype,
            ).reshape(-1)
            self.prefactor = self.calculate_prefactor()
        return self

    def get_gradients(self, features, hp, KXX, **kwargs):
        hp_deriv = {}
        n_data, m_data = len(features), len(KXX)
        if "prefactor" in hp:
            hp_deriv["prefactor"] = asarray(
                [
                    2.0
                    * exp(2.0 * self.hp["prefactor"][0])
                    * self.add_regularization(KXX, n_data, overwrite=False)
                ],
            )
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

    def calculate_prefactor(self, features=None, targets=None, **kwargs):
        return exp(2.0 * self.hp["prefactor"][0])
