from numpy import array, asarray, log, ndarray, sum as sum_, sqrt
from .pdistributions import Prior_distribution


class Gen_normal_prior(Prior_distribution):
    """
    Independent Generalized Normal prior distribution used for each type
    of hyperparameters in log-space.
    If the type of the hyperparameter is multi dimensional (H),
    it is given in the axis=-1.
    If multiple values (M) of the hyperparameter(/s)
    are calculated simultaneously, it has to be in a (M,H) array.
    """

    def __init__(self, mu=0.0, s=10.0, v=2, dtype=float, **kwargs):
        """
        Initialization of the prior distribution.

        Parameters:
            mu: float or (H) array
                The mean of the generalized normal distribution.
            std: float or (H) array
                The scale of the generalized normal distribution.
            v: float or (H) array
                The shape or magnitude of the generalized normal distribution.
            dtype: type
                The data type of the arrays.
        """
        self.update_arguments(mu=mu, s=s, v=v, dtype=dtype, **kwargs)

    def ln_pdf(self, x):
        ln_pdf = (
            -(((x - self.mu) / self.s) ** (2 * self.v))
            - log(self.s)
            + log(0.52)
        )
        if self.nosum:
            return ln_pdf
        return sum_(ln_pdf, axis=-1)

    def ln_deriv(self, x):
        return (-(2.0 * self.v) * ((x - self.mu) ** (2 * self.v - 1))) / (
            self.s ** (2 * self.v)
        )

    def set_dtype(self, dtype, **kwargs):
        super().set_dtype(dtype, **kwargs)
        if hasattr(self, "mu") and isinstance(self.mu, ndarray):
            self.mu = asarray(self.mu, dtype=self.dtype)
        if hasattr(self, "std") and isinstance(self.std, ndarray):
            self.std = asarray(self.std, dtype=self.dtype)
        if hasattr(self, "v") and isinstance(self.v, ndarray):
            self.v = asarray(self.v, dtype=self.dtype)
        return self

    def update_arguments(self, mu=None, s=None, v=None, dtype=None, **kwargs):
        """
        Update the object with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            mu: float or (H) array
                The mean of the generalized normal distribution.
            std: float or (H) array
                The scale of the generalized normal distribution.
            v: float or (H) array
                The shape or magnitude of the generalized normal distribution.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the arguments for the parent class
        super().update_arguments(
            dtype=dtype,
        )
        if mu is not None:
            if isinstance(mu, (float, int)):
                self.mu = mu
            else:
                self.mu = array(mu, dtype=self.dtype).reshape(-1)
        if s is not None:
            if isinstance(s, (float, int)):
                self.s = s
            else:
                self.s = array(s, dtype=self.dtype).reshape(-1)
        if v is not None:
            if isinstance(v, (float, int)):
                self.v = v
            else:
                self.v = array(v, dtype=self.dtype).reshape(-1)
        if (
            isinstance(self.mu, (float, int))
            and isinstance(self.s, (float, int))
            and isinstance(self.v, (float, int))
        ):
            self.nosum = True
        else:
            self.nosum = False
        return self

    def mean_var(self, mean, var):
        return self.update_arguments(mu=mean, s=sqrt(var / 0.32))

    def min_max(self, min_v, max_v):
        mu = (max_v + min_v) / 2.0
        return self.update_arguments(
            mu=mu,
            s=sqrt(2.0 / 0.32) * (max_v - mu),
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(mu=self.mu, s=self.s, v=self.v, dtype=self.dtype)
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
