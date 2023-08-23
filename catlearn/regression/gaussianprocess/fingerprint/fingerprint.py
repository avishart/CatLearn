import numpy as np
from .fingerprintobject import FingerprintObject

class Fingerprint:
    def __init__(self,reduce_dimensions=True,use_derivatives=True,mic=True,**kwargs):
        """ Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
            Parameters:
                reduce_dimensions: bool
                    Whether to reduce the fingerprint space if constrains are used.
                use_derivatives: bool
                    Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
                mic: bool
                    Minimum Image Convention (Shortest distances when periodic boundary is used).
        """
        self.reduce_dimensions=reduce_dimensions
        self.use_derivatives=use_derivatives
        self.mic=mic
        
    def __call__(self,atoms,**kwargs):
        """ Convert atoms to fingerprint and return the fingerprint object 
            Parameters:
                atoms: ASE Atoms
                    The ASE Atoms object that are converted to a fingerprint.
        """
        not_masked=self.get_constrains(atoms)
        vector,derivative=self.make_fingerprint(atoms,not_masked=not_masked,**kwargs)  
        if self.use_derivatives:  
            return FingerprintObject(vector=vector,derivative=derivative)
        return FingerprintObject(vector=vector,derivative=None)
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the fingerprint "
        raise NotImplementedError()
        
    def get_constrains(self,atoms):
        " Get the indicies of the atoms that does not have fixed constrains "
        not_masked=list(range(len(atoms)))
        if not self.reduce_dimensions:
            return not_masked
        constraints=atoms.constraints
        if len(constraints)>0:
            from ase.constraints import FixAtoms
            index_mask=np.array([c.get_indices() for c in constraints if isinstance(c,FixAtoms)]).flatten()
            index_mask=sorted(list(set(index_mask)))
            return [i for i in not_masked if i not in index_mask]
        return not_masked
    
    def copy(self):
        " Copy the Fingerprint. "
        return self.__class__(reduce_dimensions=self.reduce_dimensions,use_derivatives=self.use_derivatives,mic=self.mic)
    
    def __repr__(self):
        return "Fingerprint(reduce_dimensions={},use_derivatives={},mic={})".format(self.reduce_dimensions,self.use_derivatives,self.mic)
        