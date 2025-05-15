from .constant import Prior_constant


class Prior_first(Prior_constant):
    def __init__(self, yp=0.0, add=0.0, dtype=float, **kwargs):
        """
        The prior mean of the targets.
        The prior mean is used as a baseline of the target values.
        The prior mean is the first target value if given else it is 0.
        A value can be added to the constant.

        Parameters:
            yp: float
                The prior mean constant
            add: float
                A value added to the found prior mean from data.
            dtype: type
                The data type of the arrays.
        """
        self.update_arguments(yp=yp, add=add, dtype=dtype, **kwargs)

    def update(self, features, targets, **kwargs):
        self.update_arguments(yp=targets.item(0))
        return self
