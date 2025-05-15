from numpy import matmul
from numpy.linalg import svd
from .factorized_likelihood import (
    FactorizedLogLikelihood,
    VariableTransformation,
)


class FactorizedLogLikelihoodSVD(FactorizedLogLikelihood):
    def __init__(
        self,
        get_prior_mean=False,
        ngrid=80,
        bounds=VariableTransformation(),
        noise_optimizer=None,
        dtype=float,
        **kwargs
    ):
        """
        The factorized log-likelihood objective function that is used
        to optimize the hyperparameters.
        A SVD is performed to get the eigenvalues.
        The relative-noise hyperparameter can be searched from
        a single eigendecomposition for each length-scale hyperparameter.

        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean
                in the solution.
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
            ngrid=ngrid,
            bounds=bounds,
            noise_optimizer=noise_optimizer,
            dtype=dtype,
            **kwargs
        )

    def get_eig(self, model, X, Y):
        "Calculate the eigenvalues"
        # Calculate the kernel with and without noise
        KXX, n_data = self.kxx_corr(model, X)
        # SVD
        U, D, Vt = svd(KXX, hermitian=True)
        # Subtract the prior mean to the training target
        Y_p = self.y_prior(X, Y, model, D=D, U=U)
        UTY = matmul(Vt, Y_p).reshape(-1) ** 2
        return D, U, Y_p, UTY, KXX, n_data
