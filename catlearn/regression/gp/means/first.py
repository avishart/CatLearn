from .constant import Prior_constant


class Prior_first(Prior_constant):
    """
    The prior mean of the targets.
    The prior mean is used as a baseline of the target values.
    The prior mean is the first target value if given else it is 0.
    A value can be added to the constant.
    """

    def update(self, features, targets, **kwargs):
        self.update_arguments(yp=targets.item(0))
        return self
