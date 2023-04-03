import numpy as np
import copy

class FingerprintObject:
    def __init__(self,vector=None,derivative=None):
        """ Fingerprint object class that has the fingerprint vector for anatoms object.
            Parameters:
                vector: (N) array
                    Fingerprint vector generated from a Fingerprint constructer.
                derivative: (D,N) array (optional)
                    Fingerprint derivative wrt. atoms cartesian coordinates.
        """
        self.vector=vector.copy()
        if derivative is None:
            self.derivative=None
        else:
            self.derivative=derivative.copy()
    
    def get_vector(self):
        " Get the fingerprint vector "
        return self.vector.copy()
    
    def get_derivatives(self,d=None):
        " Get the derivative of the fingerprint wrt the cartesian coordinates"
        if d is None:
            return self.derivative.copy()
        return self.derivative[:,d].copy()
    
    def get_derivative_dimension(self):
        " Get the dimensions of the cartesian coordinates used for calculating the derivative "
        return len(self.derivative[0])
    
    def __repr__(self):
        return str(self.vector)
        