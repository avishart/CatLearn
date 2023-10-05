import numpy as np
from .fingerprint import Fingerprint

class Cartesian(Fingerprint):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,mic=True,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        The cartesian coordinate fingerprint is generated.
        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
        """
        # Set the arguments
        super().__init__(reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         mic=mic,
                         **kwargs)
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the cartesian coordinates fingerprint "
        vector=atoms[not_masked].get_positions().reshape(-1)
        if self.use_derivatives:
            derivative=np.identity(len(vector))
        else:
            derivative=None
        return vector,derivative
