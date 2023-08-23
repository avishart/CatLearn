import numpy as np
from .constant import Prior_constant

class Prior_median(Prior_constant):
    """ The prior uses a baseline of the target values as the median target value. """

    def update(self,X,Y,**kwargs):
        "The prior will use the median of the target values"
        self.set_parameters(yp=np.median(Y[:,0]))
        return self
    
    def __repr__(self):
        return "Prior_median(yp={:.4f},add={:.4f})".format(self.yp,self.add) 