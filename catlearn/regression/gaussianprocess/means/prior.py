class Prior:
    def __init__(self,**kwargs):
        """
        The prior mean of the targets. 
        The prior mean is used as a baseline of the target values.
        """
        self.update_arguments(**kwargs)
    
    def get(self,X,Y,get_derivatives=True,**kwargs):
        """
        Get the prior mean of the targets.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array or (N,1+D) array
                Training targets with N data points.
                If get_derivatives=True, the training targets is in first column and derivatives is in the next columns.
            get_derivatives: bool
                Whether to give the prior mean of the derivatives of targets.
        """
        raise NotImplementedError()
    
    def update(self,X,Y,**kwargs):
        """
        Update the prior mean with the given data.
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array or (N,1+D) array
                Training targets with N data points.
                If get_derivatives=True, the training targets is in first column and derivatives is in the next columns.
        Returns:
            self: The updated object itself.
        """
        self.update_arguments()
        return self
    
    def get_parameters(self,**kwargs):
        """
        Get the prior mean parameters.
        Returns:
            dict: A dictionary with the parameters used in the prior mean.
        """
        return dict()
    
    def update_arguments(self,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Returns:
            self: The updated object itself.
        """
        return self
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict()
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    
    def copy(self):
        " Copy the object. "
        # Get all arguments
        arg_kwargs,constant_kwargs,object_kwargs=self.get_arguments()
        # Make a clone
        clone=self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key,value in constant_kwargs.items():
                clone.__dict__[key]=value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key,value in object_kwargs.items():
                clone.__dict__[key]=value.copy()
        return clone
    
    def __repr__(self):
        arg_kwargs=self.get_arguments()[0]
        str_kwargs=",".join([f"{key}={value}" for key,value in arg_kwargs.items()])
        return "{}({})".format(self.__class__.__name__,str_kwargs)
