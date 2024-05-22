import numpy as np
from .fingerprint import Fingerprint

class Cartesian(Fingerprint):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        The cartesian coordinate fingerprint is generated.
        
        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
        """
        # Set the arguments
        super().__init__(reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         **kwargs)
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the cartesian coordinates fingerprint "
        vector=(atoms.get_positions()[not_masked]).reshape(-1)
        if self.use_derivatives:
            derivative=np.identity(len(vector))
        else:
            derivative=None
        return vector,derivative
