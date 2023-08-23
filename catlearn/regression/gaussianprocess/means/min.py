import numpy as np
from .constant import Prior_constant

class Prior_min(Prior_constant):
    """ The prior uses a baseline of the target values as the minimum target value. """
    
    def update(self,X,Y,**kwargs):
        "The prior will use the minimum of the target values"
        self.set_parameters(yp=np.min(Y[:,0]))
        return self
    
    def __repr__(self):
        return "Prior_min(yp={:.4f},add={:.4f})".format(self.yp,self.add) 