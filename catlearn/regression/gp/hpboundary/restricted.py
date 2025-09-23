from numpy import array, asarray, finfo, full, log, sqrt
from .length import LengthBoundaries


class RestrictedBoundaries(LengthBoundaries):
    """
    Boundary conditions for the hyperparameters with educated guess for
    the length-scale and relative-noise hyperparameters.
    Machine precisions are used as default boundary conditions for
    the rest of the hyperparameters not given in the dictionary.
    """

    def make_bounds(self, model, X, Y, parameters, parameters_set, **kwargs):
        eps_lower, eps_upper = self.get_boundary_limits()
        self.get_use_derivatives(model)
        bounds = {}
        for para in parameters_set:
            if para == "length":
                bounds[para] = self.length_bound(X, parameters.count(para))
            elif para == "noise":
                if "noise_deriv" in parameters_set:
                    bounds[para] = self.noise_bound(
                        Y[:, 0:1],
                        eps_lower=eps_lower,
                    )
                else:
                    bounds[para] = self.noise_bound(Y, eps_lower=eps_lower)
            elif para == "noise_deriv":
                bounds[para] = self.noise_bound(Y[:, 1:], eps_lower=eps_lower)
            elif para in self.bounds_dict:
                bounds[para] = array(self.bounds_dict[para], dtype=self.dtype)
            else:
                bounds[para] = full(
                    (parameters.count(para), 2),
                    [eps_lower, eps_upper],
                    dtype=self.dtype,
                )
        return bounds

    def noise_bound(
        self,
        Y,
        eps_lower=None,
        **kwargs,
    ):
        """
        Get the minimum and maximum ranges of the noise in
        the educated guess regime within a scale.
        """
        if eps_lower is None:
            eps_lower = 10.0 * sqrt(2.0 * finfo(self.dtype).eps)
        n_max = len(Y.reshape(-1)) * self.scale
        if self.use_log:
            return asarray([[eps_lower, log(n_max)]], dtype=self.dtype)
        return asarray([[eps_lower, n_max]], dtype=self.dtype)
