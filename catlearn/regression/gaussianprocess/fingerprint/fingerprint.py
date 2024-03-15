import numpy as np
from ase.constraints import FixAtoms
from .fingerprintobject import FingerprintObject

class Fingerprint:
    def __init__(self,reduce_dimensions=True,use_derivatives=True,mic=True,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
        """
        # Set the arguments
        self.update_arguments(reduce_dimensions=reduce_dimensions,
                              use_derivatives=use_derivatives,
                              mic=mic,
                              **kwargs)
        
    def __call__(self,atoms,**kwargs):
        """ 
        Convert atoms to fingerprint and return the fingerprint object 
        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object that are converted to a fingerprint.
        Returns:
            FingerprintObject: Object with the fingerprint array and its derivatives if requested.
        """
        # Get the constraints from ASE Atoms
        not_masked=self.get_constraints(atoms)
        # Calculate the fingerprint and its derivatives if requested 
        vector,derivative=self.make_fingerprint(atoms,not_masked=not_masked,**kwargs)
        # Make the fingerprint object and store the arrays within
        if self.use_derivatives:  
            return FingerprintObject(vector=vector,derivative=derivative)
        return FingerprintObject(vector=vector,derivative=None)
    
    def get_use_derivatives(self):
        " Get whether the derivatives of the targets are used. "
        return self.use_derivatives

    def get_reduce_dimensions(self):
        " Get whether the reduction of the fingerprint space is used if constrains are used. "
        return self.reduce_dimensions
    
    def update_arguments(self,reduce_dimensions=None,use_derivatives=None,mic=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
        Returns:
            self: The updated object itself.
        """
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        if mic is not None:
            self.mic=mic
        return self
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the fingerprint "
        raise NotImplementedError()
        
    def get_constraints(self,atoms,**kwargs):
        """
        Get the indicies of the atoms that does not have fixed constraints.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            list: A list of indicies for the moving atoms if constraints are used. 
        """
        not_masked=list(range(len(atoms)))
        if not self.reduce_dimensions:
            return not_masked
        constraints=atoms.constraints
        if len(constraints)>0:
            index_mask=np.concatenate([c.get_indices() for c in constraints if isinstance(c,FixAtoms)])
            index_mask=set(index_mask)
            return list(set(not_masked).difference(index_mask))
        return not_masked
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(reduce_dimensions=self.reduce_dimensions,
                        use_derivatives=self.use_derivatives,
                        mic=self.mic)
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
