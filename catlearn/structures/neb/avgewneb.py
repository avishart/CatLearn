from numpy import where
from .ewneb import EWNEB


class AvgEWNEB(EWNEB):
    """
    The average energy-weighted Nudged Elastic Band method implementation.
    The energy-weighted method uses energy weighting to calculate the
    spring constants.
    The average weigting for both ends of the spring are used
    instead of the forward energy weighting.
    """

    def get_spring_constants(self, **kwargs):
        # Get the spring constants
        energies = self.get_energies()
        # Get the reference energy
        if self.use_minimum:
            e0 = min([energies[0], energies[-1]])
        else:
            e0 = max([energies[0], energies[-1]])
        # Get the maximum energy
        emax = energies.max()
        # Calculate the weighted spring constants
        k_l = self.k * self.kl_scale
        if e0 < emax:
            a = (emax - energies) / (emax - e0)
            a = where(a < 1.0, a, 1.0)
            a = 0.5 * (a[1:] + a[:-1])
            k = ((1.0 - a) * self.k) + (a * k_l)
        else:
            k = k_l
        return k
