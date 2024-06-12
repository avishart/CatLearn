import numpy as np

class FingerprintObject:
    def __init__(self,vector,derivative=None,**kwargs):
        """ 
        Fingerprint object class that has the fingerprint vector for an atoms object.
        The derivatives wrt. to the cartesian coordinates can also be saved.

        Parameters:
            vector: (N) array
                Fingerprint vector generated from a Fingerprint constructer.
            derivative: (N,D) array (optional)
                Fingerprint derivative wrt. atoms cartesian coordinates.
        """
        self.vector=vector.copy()
        if derivative is None:
            self.derivative=None
        else:
            self.derivative=derivative.copy()
    
    def get_vector(self,**kwargs):
        " Get the fingerprint vector. "
        return self.vector.copy()
    
    def get_derivatives(self,d=None,**kwargs):
        " Get the derivative of the fingerprint wrt. the cartesian coordinates. "
        if self.derivative is None:
            return None
        if d is None:
            return self.derivative.copy()
        return self.derivative[:,d].copy()
    
    def get_derivative_dimension(self,**kwargs):
        " Get the dimensions of the cartesian coordinates used for calculating the derivative. "
        return len(self.derivative[0])
    
    def copy(self):
        " Copy the Fingerprint object. "
        return self.__class__(vector=self.vector,derivative=self.derivative)

    def __len__(self):
        " Get len of the vector. "
        return len(self.vector)
    
    def __repr__(self):
        return str(self.vector)
        