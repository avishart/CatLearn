from numpy import full, zeros
from .prior import Prior


class Prior_constant(Prior):
    """
    The prior mean of the targets.
    The prior mean is used as a baseline of the target values.
    The prior mean is a constant from the target values
    if given else it is 0.
    A value can be added to the constant.
    """

    def __init__(self, yp=0.0, add=0.0, dtype=float, **kwargs):
        """
        Initialize the prior mean.

        Parameters:
            yp: float
                The prior mean constant
            add: float
                A value added to the found prior mean from data.
            dtype: type
                The data type of the arrays.
        """
        self.update_arguments(yp=yp, add=add, dtype=dtype, **kwargs)

    def get(self, features, targets, get_derivatives=True, **kwargs):
        if get_derivatives:
            yp = zeros(targets.shape, dtype=self.dtype)
            yp[:, 0] = self.prior_mean
            return yp
        return full(targets.shape, self.prior_mean, dtype=self.dtype)

    def get_parameters(self, **kwargs):
        return dict(yp=self.yp, add=self.add)

    def update_arguments(self, yp=None, add=None, dtype=None, **kwargs):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            yp: float
                The prior mean constant
            add: float
                A value added to the found prior mean from data.

        Returns:
            self: The updated object itself.
        """
        super().update_arguments(dtype=dtype)
        if add is not None:
            self.add = add
        if yp is not None:
            self.yp = yp
        self.prior_mean = self.yp + self.add
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(yp=self.yp, add=self.add, dtype=self.dtype)
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
