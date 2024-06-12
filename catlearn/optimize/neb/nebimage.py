from ...regression.gaussianprocess.calculator.copy_atoms import copy_atoms

class NEBImage:
    def __init__(self,atoms):
        """
        An image for NEB as a wrapper for the Atoms instance.
        The calculated results are stored within so multiple calculations can be avoided. 

        Parameters:
            atoms : Atoms instance.
                The Atoms instance with a calculator.
        """
        self.atoms=atoms
        self.cell=self.atoms.cell
        self.pbc=self.atoms.pbc
        self.reset()

    def get_positions(self,*args,**kwargs):
        return self.atoms.get_positions(*args,**kwargs)

    def set_positions(self,*args,**kwargs):
        output=self.atoms.set_positions(*args,**kwargs)
        self.reset()
        return output

    def get_property(self,name,allow_calculation=True,**kwargs):
        """ 
        Get or calculate the requested property. 

        Parameters:
            name : str
                The name of the requested property.
            allow_calculation : bool
                Whether the property is allowed to be calculated.

        Returns:
            float or list: The requested property.
        """
        if (self.atoms_saved.calc is not None) and (name in self.atoms_saved.calc.results):
            return self.atoms_saved.calc.get_property(name,allow_calculation=True,**kwargs)
        output=self.atoms.calc.get_property(name,atoms=self.atoms,allow_calculation=allow_calculation,**kwargs)
        self.store_results()
        return output
    
    def get_potential_energy(self,*args,**kwargs):
        if (self.atoms_saved.calc is not None) and ('energy' in self.atoms_saved.calc.results):
            return self.atoms_saved.get_potential_energy(*args,**kwargs)
        energy=self.atoms.get_potential_energy(*args,**kwargs)
        self.store_results()
        return energy
    
    def get_forces(self,*args,**kwargs):
        if (self.atoms_saved.calc is not None) and ('force' in self.atoms_saved.calc.results):
            return self.atoms_saved.get_forces(*args,**kwargs)
        force=self.atoms.get_forces(*args,**kwargs)
        self.store_results()
        return force
    
    def get_atomic_numbers(self):
        return self.atoms.get_atomic_numbers()
    
    def get_cell(self):
        return self.atoms.get_cell()
    
    def get_tags(self):
        return self.atoms.get_tags()

    def store_results(self,**kwargs):
        """
        Store the calculated results.
        """
        self.atoms_saved=copy_atoms(self.atoms)
        self.calc=self.atoms_saved.calc
        return self.atoms_saved

    def reset(self,**kwargs):
        """ 
        Reset the stored properties. 
        """
        self.atoms_saved=self.atoms.copy()
        self.calc=None
        return self
    
    def __len__(self):
        return len(self.atoms)
    
    def copy(self):
        " Copy and get the Atoms instance. "
        return self.atoms.copy()
    