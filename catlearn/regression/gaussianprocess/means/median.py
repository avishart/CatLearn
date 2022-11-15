import numpy as np
from .constant import Prior_constant

class Prior_median(Prior_constant):
    def __init__(self,yp=0):
        "The prior uses a baseline of the target values if given else it is at 0"
        Prior_constant.__init__(self,yp)
    
    def update(self,X,Y):
        "The prior will use the median of the target values"
        self.dim=len(Y[0])
        self.yp=np.median(Y[:,0])
        return self.yp
