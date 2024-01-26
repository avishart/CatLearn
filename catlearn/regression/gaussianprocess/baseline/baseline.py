import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from ase.constraints import FixAtoms

class Baseline_calculator(Calculator):
    implemented_properties=['energy','forces']
    nolabel=True
    
    def __init__(self,mic=True,reduce_dimensions=True,**kwargs):
        """ 
        A baseline calculator for ASE atoms object. 
        It uses a flat baseline with zero energy and forces.  

        Parameters: 
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
        """
        super().__init__()
        self.update_arguments(mic=mic,
                              reduce_dimensions=reduce_dimensions,
                              **kwargs)
    
    def calculate(self,atoms=None,properties=['energy','forces'],system_changes=all_changes):
        """
        Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()*.
        """
        # Atoms object.
        super().calculate(atoms,properties,system_changes)
        # Obtain energy and forces for the given structure:
        if 'forces' in properties:
            energy,forces=self.get_energy_forces(atoms)
            self.results['forces']=forces
        else:
            energy=self.get_energy(atoms)
        self.results['energy']=energy
        pass

    def update_arguments(self,mic=None,reduce_dimensions=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        
        Parameters: 
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
        Returns:
            self: The updated object itself.
        """
        if mic is not None:
            self.mic=mic
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        return self

    def get_energy(self,atoms):
        " Get only the energy. "
        return 0.0

    def get_energy_forces(self,atoms):
        " Get the energy and forces. "
        return 0.0,np.zeros((len(atoms),3))
    
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
        arg_kwargs=dict(mic=self.mic,
                        reduce_dimensions=self.reduce_dimensions)
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
