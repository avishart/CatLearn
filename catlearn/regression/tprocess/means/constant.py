import numpy as np
import copy

class Prior_constant:
    def __init__(self,yp=0.0,add=0.0,**kwargs):
        "The prior uses a baseline of the target values if given else it is at 0"
        self.yp=yp+add
        self.add=add
        self.dim=1
    
    def get(self,X,get_derivatives=True):
        "Give the baseline value of the target"
        if get_derivatives:
            return np.array([[self.yp]+[0]*(self.dim-1)]*len(X))
        return np.array([[self.yp]]*len(X))
    
    def update(self,X,Y=np.array([[0]])):
        "The prior will use a fixed value"
        self.dim=len(Y[0])
        return self.yp

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return "Prior_constant(yp={:.4f})".format(self.yp)  
