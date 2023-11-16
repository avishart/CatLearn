import numpy as np

class HPBoundaries:
    def __init__(self,bounds_dict={},scale=1.0,log=True,**kwargs):
        """ 
        Boundary conditions for the hyperparameters.
        A dictionary with boundary conditions of the hyperparameters can be given as an argument.
        Machine precisions are used as boundary conditions for the hyperparameters not given in the dictionary.
        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
        """
        self.update_arguments(bounds_dict=bounds_dict,
                              scale=scale,
                              log=log,
                              **kwargs)

    def update_bounds(self,model,X,Y,parameters,**kwargs):
        """ 
        Create and update the boundary conditions for the hyperparameters. 
        Parameters:
            model : Model
                The Machine Learning Model with kernel and prior that are optimized.
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            parameters : (H) list of strings
                A list of names of the hyperparameters.
        Returns:
            self : The object itself.
        """
        # Update parameters
        self.make_parameters_set(parameters)
        # Make bounds (Parameters and keys in bounds_dict have to be the same)
        self.bounds_dict=self.make_bounds(model,X,Y,parameters,self.parameters_set)
        return self

    def get_bounds(self,parameters=None,array=False,**kwargs):
        """ 
        Get the boundary conditions of the hyperparameters. 
        Parameters :
            parameters : list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            array : bool
                Whether to get an array or a dictionary as output.
        Returns:
            (H,2) array : The boundary conditions as an array if array=True.
            or
            dict : A dictionary of the boundary conditions. 
        """
        # Make the sorted unique hyperparameters if they are given
        parameters_set=self.get_parameters_set(parameters=parameters)
        # Make the boundary conditions for the given hyperparameters
        if array:
            return np.concatenate([self.bounds_dict[para] for para in parameters_set],axis=0)
        return {para:self.bounds_dict[para].copy() for para in parameters_set}
    
    def get_hp(self,parameters=None,array=False,**kwargs):
        """ 
        Get the guess of the hyperparameters. 
        The mean of the boundary conditions in log-space is used as the guess. 
        Parameters:
            parameters : list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            array : bool
                Whether to get an array or a dictionary as output.
        Returns:
            (H) array : The guesses of the hyperparameters as an array if array=True.
            or
            dict : A dictionary of the guesses of the hyperparameters. 
        """
        # Make the sorted unique hyperparameters if they are given
        parameters_set=self.get_parameters_set(parameters=parameters)
        if self.log:
            if array:
                return np.concatenate([np.mean(self.bounds_dict[para],axis=1) for para in parameters_set])
            return {para:np.mean(self.bounds_dict[para],axis=1) for para in parameters_set}
        if array:
            return np.concatenate([np.exp(np.mean(np.log(self.bounds_dict[para]),axis=1)) for para in parameters_set])
        return {para:np.exp(np.mean(np.log(self.bounds_dict[para]),axis=1)) for para in parameters_set}
    
    def make_lines(self,parameters=None,ngrid=80,**kwargs):
        """ 
        Make grid in each dimension of the hyperparameters from the boundary conditions.
        Parameters:
            parameters : list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            ngrid : int or (H) list
                An integer or a list with number of grid points in each dimension.
        Returns:
            (H,) list : A list with grid points for each (H) hyperparameters.
        """
        bounds=self.get_bounds(parameters=parameters,array=True)
        if isinstance(ngrid,(int,float)):
            ngrid=[int(ngrid)]*len(bounds)
        return [np.linspace(bound[0],bound[1],ngrid[b]) for b,bound in enumerate(bounds)]
    
    def make_single_line(self,parameter,ngrid=80,i=0,**kwargs):
        """ 
        Make grid in one dimension of the hyperparameters from the boundary conditions.
        Parameters:
            parameters : str
                A string of the hyperparameter name.
            ngrid : int
                An integer with number of grid points in each dimension.
            i : int
                The index of the hyperparameter used if multiple hyperparameters of the same type exist. 
        Returns: 
            (ngrid) array : A grid of ngrid points for the given hyperparameter.
        """
        if not isinstance(ngrid,(int,float)):
            ngrid=ngrid[int(self.parameters.index(parameter)+i)]
        bound=self.bounds_dict[parameter][i]
        return np.linspace(bound[0],bound[1],int(ngrid))
    
    def sample_thetas(self,parameters=None,npoints=50,**kwargs):
        """ 
        Sample hyperparameters from the boundary conditions. 
        Parameters:
            parameters : list of str or None
                A list of the specific used hyperparameter names as strings.
                If parameters=None, then the stored hyperparameters are used.
            npoints : int
                Number of points to sample.   
        Returns:
            (npoints,H) array : An array with sampled hyperparameters. 
        """
        bounds=self.get_bounds(parameters=parameters,array=True)
        return np.random.uniform(low=bounds[:,0],high=bounds[:,1],size=(int(npoints),len(bounds)))
    
    def update_arguments(self,bounds_dict=None,scale=None,log=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
        Returns:
            self: The updated object itself.
        """
        if bounds_dict is not None:
            self.initiate_bounds_dict(bounds_dict)
        if scale is not None:
            self.scale=scale
        if log is not None:
            self.log=log
        return self
    
    def make_bounds(self,model,X,Y,parameters,parameters_set,**kwargs):
        " Make the boundary conditions with educated guesses of the length-scale hyperparamer(s). "
        eps_lower,eps_upper=self.get_boundary_limits()
        bounds={}
        for para in parameters_set:
            if para in self.bounds_dict:
                bounds[para]=self.bounds_dict[para].copy()
            else:
                bounds[para]=np.full((parameters.count(para),2),[eps_lower,eps_upper])
        return bounds
    
    def initiate_bounds_dict(self,bounds_dict,**kwargs):
        " Make and store the hyperparameter types and the dictionary with hyperparameter bounds. "
        # Copy the boundary condition values
        self.bounds_dict={key:np.array(value) for key,value in bounds_dict.items()}
        if 'correction' in self.bounds_dict.keys():
            self.bounds_dict.pop('correction')
        # Extract the hyperparameter names
        self.parameters_set=sorted(bounds_dict.keys())
        self.parameters=sum([[para]*len(bounds_dict[para]) for para in self.parameters_set],[])
        return self
    
    def make_parameters_set(self,parameters,**kwargs):
        " Make and store the hyperparameters types. "
        parameters=list(parameters)
        if 'correction' in parameters:
            parameters.remove('correction')
        self.parameters=sorted(parameters)
        self.parameters_set=sorted(set(parameters))
        return self.parameters_set

    def get_parameters_set(self,parameters=None,**kwargs):
        " Make the sorted unique hyperparameters if they are given else get the stored. "
        # Get the sorted and unique stored hyperparameter sets
        if parameters is None:
            return self.parameters_set.copy()
        # Make the sorted unique hyperparameters if they are given
        return sorted(set(parameters))
    
    def get_n_parameters(self,parameters=None,**kwargs):
        " Get the number of hyperparameters used or stored. "
        if parameters is None:
            return len(self.parameters)
        return len(parameters)
    
    def get_boundary_limits(self,**kwargs):
        " Get the machine precision limits for the hyperparameters. "
        eps_lower=10*np.sqrt(2.0*np.finfo(float).eps)/self.scale
        if self.log:
            eps_lower=np.log(eps_lower)
            return eps_lower,-eps_lower
        return eps_lower,1.0/eps_lower
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(bounds_dict=self.bounds_dict,
                        scale=self.scale,
                        log=self.log)
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
