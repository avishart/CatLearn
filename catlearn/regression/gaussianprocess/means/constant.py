import numpy as np
from .prior import Prior

class Prior_constant(Prior):
    def __init__(self,yp=0.0,add=0.0,**kwargs):
        """
        The prior mean of the targets. 
        The prior mean is used as a baseline of the target values.
        The prior mean is a constant from the target values if given else it is 0. 
        A value can be added to the constant.
        Parameters:
            yp : float
                The prior mean constant
            add : float
                A value added to the found prior mean from data.
        """
        self.update_arguments(yp=yp,add=add,**kwargs)

    def get(self,X,Y,get_derivatives=True,**kwargs):
        if get_derivatives:
            yp=np.zeros(Y.shape)
            yp[:,0]=self.yp
            return yp
        return np.full(Y.shape,self.yp)
    
    def get_parameters(self,**kwargs):
        return dict(yp=self.yp,add=self.add)
    
    def update_arguments(self,yp=None,add=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            yp : float
                The prior mean constant
            add : float
                A value added to the found prior mean from data.
        Returns:
            self: The updated object itself.
        """
        if add is not None:
            self.add=add
        if yp is not None:
            self.yp=yp+self.add
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(yp=self.yp,add=self.add)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
