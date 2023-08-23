import numpy as np
from .prior import Prior

class Prior_constant(Prior):
    def __init__(self,yp=0.0,add=0.0,**kwargs):
        """ The prior uses a constant baseline of the target values if given else it is 0. 
            A value can be added to the constant. """
        self.set_parameters(yp=yp,add=add,**kwargs)
    
    def get(self,X,Y,get_derivatives=True,**kwargs):
        "Give the baseline value of the target"
        if get_derivatives:
            yp=np.zeros(Y.shape)
            yp[:,0]=self.yp
            return yp
        return np.full(Y.shape,self.yp)
    
    def update(self,X,Y,**kwargs):
        "The prior will use a fixed value. "
        self.set_parameters()
        return self
    
    def set_parameters(self,yp=None,add=None,**kwargs):
        " Set the parameters. "
        if add is not None:
            self.add=add
        if yp is not None:
            self.yp=yp+self.add
        return self
    
    def get_parameters(self):
        " Get the parameters. "
        return dict(add=self.add,yp=self.yp)
    
    def copy(self):
        " Copy the prior mean object. "
        return self.__class__(yp=self.yp,add=self.add)

    def __repr__(self):
        return "Prior_constant(yp={:.4f},add={:.4f})".format(self.yp,self.add)  
    
