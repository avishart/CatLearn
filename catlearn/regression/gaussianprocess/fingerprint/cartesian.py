import numpy as np
from .fingerprint import Fingerprint

class Cartesian(Fingerprint):
    " The cartesian coordinate fingerprint "
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the cartesian coordinates fingerprint "
        vector=atoms[not_masked].get_positions().reshape(-1)
        if self.use_derivatives:
            derivative=np.identity(len(vector))
        else:
            derivative=None
        return vector,derivative
