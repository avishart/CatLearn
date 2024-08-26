import numpy as np
from .improvedneb import ImprovedTangentNEB


class MaxEWNEB(ImprovedTangentNEB):
    def __init__(
        self,
        images,
        k=0.1,
        kl_scale=0.1,
        dE=0.01,
        climb=False,
        remove_rotation_and_translation=False,
        mic=True,
        **kwargs
    ):
        super().__init__(
            images,
            k=k,
            climb=climb,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            **kwargs
        )
        self.kl_scale = kl_scale
        self.dE = dE

    def get_parallel_forces(self, tangent, pos_p, pos_m, **kwargs):
        energies = self.get_energies()
        emax = np.max(energies)
        e0 = emax - self.dE
        k_l = self.k * self.kl_scale
        if e0 < emax:
            a = (emax - energies[:-1]) / (emax - e0)
            k = np.where(a < 1.0, (1.0 - a) * self.k + a * k_l, k_l)
        else:
            k = k_l.copy()
        forces_parallel = (k[1:] * np.linalg.norm(pos_p, axis=(1, 2))) - (
            k[:-1] * np.linalg.norm(pos_m, axis=(1, 2))
        )
        forces_parallel = forces_parallel.reshape(-1, 1, 1) * tangent
        return forces_parallel
