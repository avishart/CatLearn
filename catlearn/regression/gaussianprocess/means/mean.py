import numpy as np
from .constant import Prior_constant

class Prior_mean(Prior_constant):
    """ The prior uses a baseline of the target values as the mean target value. """

    def update(self,X,Y,**kwargs):
        "The prior will use the mean of the target values"
        self.set_parameters(yp=np.mean(Y[:,0]))
        return self
    
    def __repr__(self):
        return "Prior_mean(yp={:.4f},add={:.4f})".format(self.yp,self.add) 
