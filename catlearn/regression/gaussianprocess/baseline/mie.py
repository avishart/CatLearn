import numpy as np
from ase.data import covalent_radii
from .repulsive import Repulsion_calculator
        
class Mie_calculator(Repulsion_calculator):
    implemented_properties=['energy', 'forces']
    nolabel=True
    
    def __init__(self,mic=True,reduce_dimensions=True,r_scale=1.0,denergy=0.1,power_r=8,power_a=6,**kwargs):
        """ 
        A baseline calculator for ASE atoms object. 
        It uses the Mie potential baseline.  
        The power and the scaling of the Mie potential can be selected.

        Parameters: 
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
            r_scale : float
                The scaling of the covalent radii. 
                A smaller value will move the potential to a lower distances. 
            denergy : float 
                The dispersion energy of the potential.
            power_r : int
                The power of the potential part.
            power_a : int
                The power of the attraction part.
        """
        super().__init__(mic=mic,
                         reduce_dimensions=reduce_dimensions,
                         r_scale=r_scale,
                         denergy=denergy,
                         power_a=power_a,
                         power_r=power_r,
                         **kwargs)

    def update_arguments(self,mic=None,reduce_dimensions=None,r_scale=None,denergy=None,power_r=None,power_a=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters: 
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
            denergy : float 
                The dispersion energy of the potential.
            power_r : int
                The power of the potential part.
            power_a : int
                The power of the attraction part.
                
        Returns:
            self: The updated object itself.
        """
        if mic is not None:
            self.mic=mic
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        if r_scale is not None:
            self.r_scale=float(r_scale)
        if denergy is not None:
            self.denergy=float(denergy)
        if power_r is not None:
            self.power_r=int(power_r)
        if power_a is not None:
            self.power_a=int(power_a)
        # Calculate the r_scale powers
        self.r_scale_r=self.r_scale**self.power_r
        self.r_scale_a=self.r_scale**self.power_a
        # Calculate the normalization
        c=(self.power_r/(self.power_r-self.power_a))*((self.power_r/self.power_a)**(self.power_a/(self.power_r-self.power_a)))
        self.c0=self.denergy*c
        return self
    
    def get_energy(self,atoms,**kwargs):
        " Get the energy. "
        # Get the not fixed (not masked) atom indicies
        not_masked=np.array(self.get_constraints(atoms))
        # Get the number of atoms and the number of the fixed atoms
        n_atoms=len(atoms)
        n_nm=len(not_masked)
        i_nm=np.arange(n_nm).reshape(-1,1)
        # Get the covariance distance and the distances between atoms
        cov_dis,distances,vec_distances=self.get_cov_dis(atoms,not_masked,mic=self.mic,use_derivatives=False,i_nm=i_nm.flatten())
        # Get the potential between not masked atoms and fixed atoms
        pot1,potential_deriv=self.energy_rest(None,i_nm,not_masked,False,cov_dis,distances,vec_distances,n_atoms,n_nm)
        # Get the potential between not masked atoms and not masked atoms
        pot2,potential_deriv=self.energy_nm_nm(None,i_nm,not_masked,False,cov_dis,distances,vec_distances,n_atoms,n_nm)
        return self.c0*(pot1+pot2)
        
    def get_energy_forces(self,atoms,**kwargs):
        " Get the energy and forces. "
        # Get the not fixed (not masked) atom indicies
        not_masked=np.array(self.get_constraints(atoms))
        # Get the number of atoms and the number of the fixed atoms
        n_atoms=len(atoms)
        n_nm=len(not_masked)
        i_nm=np.arange(n_nm).reshape(-1,1)
        potential_deriv=np.zeros((n_nm,3))
        # Get the covariance distance and the distances between atoms
        cov_dis,distances,vec_distances=self.get_cov_dis(atoms,not_masked,mic=self.mic,use_derivatives=True,i_nm=i_nm.flatten())
        # Get the potential between not masked atoms and fixed atoms
        pot1,potential_deriv=self.energy_rest(potential_deriv,i_nm,not_masked,True,cov_dis,distances,vec_distances,n_atoms,n_nm)
        # Get the potential between not masked atoms and not masked atoms
        pot2,potential_deriv=self.energy_nm_nm(potential_deriv,i_nm,not_masked,True,cov_dis,distances,vec_distances,n_atoms,n_nm)
        # Make the derivatives into forces
        deriv=np.zeros((n_atoms,3))
        deriv[not_masked]=self.c0*potential_deriv
        return self.c0*(pot1+pot2),deriv.reshape(n_atoms,3)
    
    def energy_rest(self,potential_deriv,i_nm,not_masked,use_derivatives,cov_dis,distances,vec_distances,n_atoms,n_nm):
        " Get the energy contribution and their derivatives of non-fixed and fixed atoms. "
        if n_atoms!=n_nm:
            i_rest=np.setdiff1d(np.arange(n_atoms),not_masked)
            r=cov_dis[i_nm,i_rest]/distances[i_nm,i_rest]
            pot1_r=self.r_scale_r*(r**self.power_r)
            pot1_a=self.r_scale_a*(r**self.power_a)
            if use_derivatives:
                pot1_d=(-self.power_r*pot1_r)+(self.power_a*pot1_a)
                potential_deriv+=np.sum(vec_distances[i_nm,i_rest]*((pot1_d/(distances[i_nm,i_rest]**2)).reshape(n_nm,n_atoms-n_nm,1)),axis=1)
            return np.sum(pot1_r-pot1_a),potential_deriv
        return 0.0,potential_deriv
    
    def energy_nm_nm(self,potential_deriv,i_nm,not_masked,use_derivatives,cov_dis,distances,vec_distances,n_atoms,n_nm):
        " Get the fingerprints and their derivatives of non-fixed and non-fixed atoms. "
        r=cov_dis[i_nm,not_masked]/distances[i_nm,not_masked]
        pot2_r=self.r_scale_r*(r**self.power_r)
        pot2_a=self.r_scale_a*(r**self.power_a)
        if use_derivatives:
            pot2_d=(-self.power_r*pot2_r)+(self.power_a*pot2_a)
            potential_deriv+=np.sum(vec_distances[i_nm,not_masked]*((pot2_d/(distances[i_nm,not_masked]**2)).reshape(n_nm,n_nm,1)),axis=1)
        return 0.5*np.sum(pot2_r-pot2_a),potential_deriv
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mic=self.mic,
                        reduce_dimensions=self.reduce_dimensions,
                        r_scale=self.r_scale,
                        denergy=self.denergy,
                        power_a=self.power_a,
                        power_r=self.power_r,)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    