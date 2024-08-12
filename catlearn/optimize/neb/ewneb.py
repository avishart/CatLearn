import numpy as np
from .improvedneb import ImprovedTangentNEB


class EWNEB(ImprovedTangentNEB):
    def __init__(
        self,
        images,
        k=0.1,
        kl_scale=0.1,
        use_minimum=False,
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
        self.use_minimum = use_minimum

    def get_parallel_forces(self, tangent, pos_p, pos_m, **kwargs):
        energies = self.get_energies()
        if self.use_minimum:
            e0 = np.min([energies[0], energies[-1]])
        else:
            e0 = np.max([energies[0], energies[-1]])
        emax = np.max(energies)
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
