import numpy as np
from .ewneb import EWNEB


class AvgEWNEB(EWNEB):

    def get_spring_constants(self, **kwargs):
        # Get the spring constants
        energies = self.get_energies()
        # Get the reference energy
        if self.use_minimum:
            e0 = np.min([energies[0], energies[-1]])
        else:
            e0 = np.max([energies[0], energies[-1]])
        # Get the maximum energy
        emax = np.max(energies)
        # Calculate the weighted spring constants
        k_l = self.k * self.kl_scale
        if e0 < emax:
            a = (emax - energies) / (emax - e0)
            a = np.where(a < 1.0, a, 1.0)
            a = 0.5 * (a[1:] + a[:-1])
            k = ((1.0 - a) * self.k) + (a * k_l)
        else:
            k = k_l
        return k
