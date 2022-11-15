import numpy as np
import copy
from ase.calculators.singlepoint import SinglePointCalculator

class Database:
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_forces=True,negative_forces=True,use_fingerprint=True):
        """ Database of ASE atoms objects that are converted into fingerprints and targets. 
            Parameters:
                fingerprint : Fingerprint object
                    An object as a fingerprint class that convert atoms to fingerprint.
                reduce_dimensions: bool
                    Whether to reduce the fingerprint space if constrains are used.
                use_forces : bool
                    Whether to use forces in the targets.
                negative_forces : bool
                    Whether derivatives (True) or forces (False) are used.
                use_fingerprint : bool
                    Whether the kernel uses fingerprint objects (True) or arrays (False).
        """
        if fingerprint is None:
            from ..fingerprint.cartesian import Cartesian
            fingerprint=Cartesian(reduce_dimensions=reduce_dimensions,use_derivatives=use_forces)
        self.fingerprint=copy.deepcopy(fingerprint)
        self.reduce_dimensions=reduce_dimensions
        self.use_forces=use_forces
        self.negative_forces=negative_forces
        self.use_fingerprint=use_fingerprint
        self.atoms_list=[]
        self.features=[]
        self.targets=[]
        
    def add(self,atoms):
        " Add an atoms object to the database. "
        self.append(atoms)
        return copy.deepcopy(self)
    
    def add_set(self,atoms_list):
        " Add a set of atoms objects to the database. "
        for atoms in atoms_list:
            self.append(atoms)
        return copy.deepcopy(self)
    
    def append(self,atoms):
        " Append the atoms object, the fingerprint, and target to lists. "
        atoms=self.copy_atoms(atoms)
        self.atoms_list.append(atoms)
        if self.use_fingerprint:
            self.features.append(self.fingerprint(atoms))
        else:
            self.features.append(self.fingerprint(atoms).get_vector())
        self.targets.append(self.get_target(atoms,use_forces=self.use_forces,negative_forces=self.negative_forces))
        pass
        
    def get_target(self,atoms,use_forces=True,negative_forces=True):
        " Calculate the target as the energy and forces if selected. "
        e=atoms.get_potential_energy()
        if use_forces:
            not_masked=self.get_constrains(atoms)
            f=(atoms.get_forces()[not_masked]).reshape(-1)
            if negative_forces:
                return np.concatenate([[e],-f]).reshape(-1)
            return np.concatenate([[e],f]).reshape(-1)
        return np.array([e])
        
    def get_constrains(self,atoms):
        " Get the indicies of the atoms that does not have fixed constrains. "
        not_masked=list(range(len(atoms)))
        if not self.reduce_dimensions:
            return not_masked
        constraints=atoms.constraints
        if len(constraints)>0:
            from ase.constraints import FixAtoms
            index_mask=np.array([c.get_indices() for c in constraints if isinstance(c,FixAtoms)]).flatten()
            index_mask=sorted(list(set(index_mask)))
            return [i for i in not_masked if i not in index_mask]
        return not_masked
    
    def get_atoms(self):
        " Get the list of atoms in the database. "
        return self.atoms_list.copy()
    
    def get_features(self):
        " Get all the fingerprints of the atoms in the database. "
        return np.array(self.features).copy()
    
    def get_targets(self):
        " Get all the targets of the atoms in the database. "
        return np.array(self.targets).copy()

    def copy_atoms(self,atoms):
        " Copy the atoms object together with the calculated energies and forces "
        # Check that the atoms does not contain any unavailable properties
        results=atoms.calc.results.copy()
        all_properties=['energy','forces','stress','stresses','dipole','charges','magmom','magmoms','free_energy','energies']
        results={key:results[key] for key in results.keys() if key in all_properties}
        # Initialize a SinglePointCalculator to store this results
        atoms0=atoms.copy()
        atoms0.calc=SinglePointCalculator(atoms, **results)
        return atoms0

    def get_atoms_feature(self,atoms):
        " Get the feature of a single Atoms object eg. for predicting "
        if self.use_fingerprint:
            return self.fingerprint(atoms)
        return self.fingerprint(atoms).get_vector()
    
    def copy(self):
        " Copy the database. "
        return copy.deepcopy(self)
    
    def __len__(self):
        " Get the number of atoms objects in the database. "
        return len(self.atoms_list)
    
    def __repr__(self):
        if self.use_forces:
            return "Database({} Atoms objects without forces)".format(len(self.atoms_list))
        return "Database({} Atoms objects with forces)".format(len(self.atoms_list))
    
    