class Prior:
    def __init__(self,**kwargs):
        "The prior mean uses a baseline of the target values."
        self.set_parameters(**kwargs)
    
    def get(self,X,Y,get_derivatives=True,**kwargs):
        "Give the baseline value of the target"
        raise NotImplementedError()
    
    def update(self,X,Y,**kwargs):
        "Update the prior mean. "
        self.set_parameters()
        return self
    
    def set_parameters(self,**kwargs):
        " Set the parameters. "
        return self
    
    def get_parameters(self):
        " Get the parameters. "
        return dict()
    
    def copy(self):
        " Copy the prior mean object. "
        return self.__class__()

    def __repr__(self):
        return "Prior()"  
