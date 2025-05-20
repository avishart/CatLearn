from numpy import (
    array,
    asarray,
    log,
    inf,
    nan_to_num,
    ndarray,
    sum as sum_,
    sqrt,
    where,
)
from .pdistributions import Prior_distribution


class Uniform_prior(Prior_distribution):
    """
    Uniform prior distribution used for each type
    of hyperparameters in log-space.
    If the type of the hyperparameter is multi dimensional (H),
    it is given in the axis=-1.
    If multiple values (M) of the hyperparameter(/s)
    are calculated simultaneously, it has to be in a (M,H) array.
    """

    def __init__(self, start=-18.0, end=18.0, prob=1.0, dtype=float, **kwargs):
        """
        Initialization of the prior distribution.

        Parameters:
            start: float or (H) array
                The start of non-zero prior distribution value of
                the hyperparameter in log-space.
            end: float or (H) array
                The end of non-zero prior distribution value of
                the hyperparameter in log-space.
            prob: float or (H) array
                The non-zero prior distribution value.
            dtype: type
                The data type of the arrays.
        """
        self.update_arguments(
            start=start,
            end=end,
            prob=prob,
            dtype=dtype,
            **kwargs,
        )

    def ln_pdf(self, x):
        ln_0 = -log(nan_to_num(inf))
        ln_pdf = where(
            x >= self.start,
            where(x <= self.end, log(self.prob), ln_0),
            ln_0,
        )
        if self.nosum:
            return ln_pdf
        return sum_(ln_pdf, axis=-1)

    def ln_deriv(self, x):
        return 0.0 * x

    def set_dtype(self, dtype, **kwargs):
        super().set_dtype(dtype, **kwargs)
        if hasattr(self, "start") and isinstance(self.start, ndarray):
            self.start = asarray(self.start, dtype=self.dtype)
        if hasattr(self, "end") and isinstance(self.end, ndarray):
            self.end = asarray(self.end, dtype=self.dtype)
        if hasattr(self, "prob") and isinstance(self.prob, ndarray):
            self.prob = asarray(self.prob, dtype=self.dtype)
        return self

    def update_arguments(
        self,
        start=None,
        end=None,
        prob=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the object with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            start: float or (H) array
                The start of non-zero prior distribution value of
                the hyperparameter in log-space.
            end: float or (H) array
                The end of non-zero prior distribution value of
                the hyperparameter in log-space.
            prob: float or (H) array
                The non-zero prior distribution value.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the arguments for the parent class
        super().update_arguments(
            dtype=dtype,
        )
        if start is not None:
            if isinstance(start, (float, int)):
                self.start = start
            else:
                self.start = array(start, dtype=self.dtype).reshape(-1)
        if end is not None:
            if isinstance(end, (float, int)):
                self.end = end
            else:
                self.end = array(end, dtype=self.dtype).reshape(-1)
        if prob is not None:
            if isinstance(prob, (float, int)):
                self.prob = prob
            else:
                self.prob = array(prob, dtype=self.dtype).reshape(-1)
        if (
            isinstance(self.start, (float, int))
            and isinstance(self.end, (float, int))
            and isinstance(self.prob, (float, int))
        ):
            self.nosum = True
        else:
            self.nosum = False
        return self

    def mean_var(self, mean, var):
        std = sqrt(var)
        return self.update_arguments(
            start=mean - 4.0 * std,
            end=mean + 4.0 * std,
            prob=1.0 / (8.0 * std),
        )

    def min_max(self, min_v, max_v):
        return self.update_arguments(
            start=min_v,
            end=max_v,
            prob=1.0 / (max_v - min_v),
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            start=self.start,
            end=self.start,
            prob=self.prob,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
