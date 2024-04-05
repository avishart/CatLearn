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
        self.calc=atoms.calc
        self.cell=self.atoms.cell
        self.pbc=self.atoms.pbc
        self.reset()

    def get_positions(self,*args,**kwargs):
        return self.atoms.get_positions(*args,**kwargs)

    def set_positions(self,*args,**kwargs):
        self.reset()
        return self.atoms.set_positions(*args,**kwargs)

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
        if name in self.results:
            return self.results[name]
        output=self.atoms.calc.get_property(name,atoms=self.atoms,allow_calculation=allow_calculation,**kwargs)
        self.store_results(self.atoms.calc.results)
        return output
    
    def get_potential_energy(self,*args,**kwargs):
        if 'energy' in self.results:
            return self.results['energy']
        self.results['energy']=self.atoms.get_potential_energy(*args,**kwargs)
        self.store_results(self.atoms.calc.results)
        return self.results['energy']
    
    def get_forces(self,*args,**kwargs):
        if 'forces' in self.results:
            return self.results['forces'].copy()
        self.results['forces']=self.atoms.get_forces(*args,**kwargs)
        self.store_results(self.atoms.calc.results)
        return self.results['forces'].copy()
    
    def get_atomic_numbers(self):
        return self.atoms.get_atomic_numbers()

    def store_results(self,result,**kwargs):
        """
        Store the calculated results.
        """
        for key,value in result.items():
            if key not in self.results:
                if key=='energy':
                    self.results['energy']=self.atoms.get_potential_energy()
                elif key=='forces':
                    self.results['forces']=self.atoms.get_forces().copy()
                if value is None:
                    continue
                elif isinstance(value,(float,int)):
                    self.results[key]=value
                else:
                    self.results[key]=value.copy()
        return self

    def reset(self,**kwargs):
        " Reset the stored properties. "
        self.results={}
        return self
    
    def __len__(self):
        return len(self.atoms)
    
    def copy(self):
        " Copy and get the Atoms instance. "
        return self.atoms.copy()
    