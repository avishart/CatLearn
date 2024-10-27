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
        save_properties=False,
        parallel=False,
        world=None,
        **kwargs
    ):
        super().__init__(
            images,
            k=k,
            climb=climb,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mic=mic,
            save_properties=save_properties,
            parallel=parallel,
            world=world,
            **kwargs
        )
        self.kl_scale = kl_scale
        self.dE = dE

    def get_spring_constants(self, **kwargs):
        # Get the spring constants
        energies = self.get_energies()
        # Get the maximum energy
        emax = np.max(energies)
        # Calculate the reference energy
        e0 = emax - self.dE
        # Calculate the weighted spring constants
        k_l = self.k * self.kl_scale
        if e0 < emax:
            a = (emax - energies[:-1]) / (emax - e0)
            k = np.where(a < 1.0, (1.0 - a) * self.k + a * k_l, k_l)
        else:
            k = k_l
        return k
