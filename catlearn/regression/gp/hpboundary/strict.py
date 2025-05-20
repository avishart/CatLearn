from numpy import asarray, log, median, ndarray, zeros
from scipy.spatial.distance import pdist
from .educated import EducatedBoundaries


class StrictBoundaries(EducatedBoundaries):
    """
    Boundary conditions for the hyperparameters with educated guess for
    the length-scale, relative-noise, and prefactor hyperparameters.
    Stricter boundary conditions are used for the length-scale hyperparameter.
    Machine precisions are used as boundary conditions for
    other hyperparameters not given in the dictionary.
    """

    def length_bound(self, X, l_dim, **kwargs):
        """
        Get the minimum and maximum ranges of the length-scale in
        the educated guess regime within a scale.
        """
        # Get the minimum and maximum scaling factors
        exp_lower = 0.2 / self.scale
        exp_max = 4.0 * self.scale
        # Scale the convergence if derivatives of targets are used
        if self.use_derivatives:
            exp_lower = exp_lower * 0.05
        lengths = zeros((l_dim, 2), dtype=self.dtype)
        # If only one features is given then end
        if len(X) == 1:
            lengths[:, 0] = exp_lower
            lengths[:, 1] = exp_max
            if self.use_log:
                return log(lengths)
            return lengths
        # Ensure that the features are a matrix
        if not isinstance(X[0], (list, ndarray)):
            X = asarray([fp.get_vector() for fp in X], dtype=self.dtype)
        for d in range(l_dim):
            # Calculate distances
            if l_dim == 1:
                dis = pdist(X)
            else:
                d1 = d + 1
                dis = pdist(X[:, d:d1])
            dis = asarray(dis, dtype=self.dtype)
            # Calculate the maximum length-scale
            dis_max = exp_max * median(dis)
            if dis_max == 0.0:
                dis_min, dis_max = exp_lower, exp_max
            else:
                # The minimum length-scale from the nearest neighbor distance
                dis_min = exp_lower * median(self.nearest_neighbors(dis))
                if dis_min == 0.0:
                    dis_min = exp_lower
            # Transform into log-scale if specified
            lengths[d, 0], lengths[d, 1] = dis_min, dis_max
        if self.use_log:
            return log(lengths)
        return lengths
