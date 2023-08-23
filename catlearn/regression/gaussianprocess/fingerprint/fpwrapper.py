from .fingerprint import Fingerprint
import numpy as np

class Fingerprint_wrapper(Fingerprint):
    def __init__(self,fingerprint,reduce_dimensions=True,use_derivatives=True,mic=True,**kwargs):
        """ Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives by wrapping the fingerprint class of gpatom.
            Parameters:
                fingerprint: gpatom class.
                    The fingerprint class from ase-gpatom.
                reduce_dimensions: bool
                    Whether to reduce the fingerprint space if constrains are used.
                use_derivatives: bool
                    Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
                mic: bool
                    Minimum Image Convention (Shortest distances when periodic boundary is used).
        """
        super().__init__(reduce_dimensions=reduce_dimensions,use_derivatives=use_derivatives,mic=mic,**kwargs)
        self.fingerprint=fingerprint

    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the gp-atom fingerprint "
        fp=self.fingerprint(atoms,calc_gradients=self.use_derivatives,**kwargs)
        vector=fp.vector.copy()
        if self.use_derivatives:
            derivative=fp.reduce_coord_gradients().copy()
            # not_masked or constrains are not possible in ASE-GPATOM so it is enforced here
            derivative=np.concatenate(derivative[not_masked],axis=1) 
        else:
            derivative=None
        return vector,derivative
    
    def copy(self):
        " Copy the Fingerprint. "
        return self.__class__(fingerprint=self.fingerprint,reduce_dimensions=self.reduce_dimensions,use_derivatives=self.use_derivatives,mic=self.mic)
    
    def __repr__(self):
        return "Fingerprint_wrapper(fingerprint={},reduce_dimensions={},use_derivatives={},mic={})".format(self.fingerprint,self.reduce_dimensions,self.use_derivatives,self.mic)

