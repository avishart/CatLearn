import numpy as np
from ase.data import covalent_radii
from .baseline import Baseline_calculator
from ..fingerprint.geometry import get_all_distances
        
class Repulsion_calculator(Baseline_calculator):
    implemented_properties=['energy', 'forces']
    nolabel=True
    
    def __init__(self,mic=True,reduce_dimensions=True,r_scale=0.7,power=12,**kwargs):
        """ 
        A baseline calculator for ASE atoms object. 
        It uses a repulsive Lennard-Jones potential baseline.  
        The power and the scaling of the repulsive Lennard-Jones potential can be selected.

        Parameters: 
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
            r_scale : float
                The scaling of the covalent radii. 
                A smaller value will move the repulsion to a lower distances. 
            power : int
                The power of the repulsion.
        """
        super().__init__(mic=mic,
                         reduce_dimensions=reduce_dimensions,
                         r_scale=r_scale,
                         power=power,
                         **kwargs)

    def update_arguments(self,mic=None,reduce_dimensions=None,r_scale=None,power=None,**kwargs):
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
        if r_scale is not None:
            self.r_scale=r_scale
        if power is not None:
            self.power=int(power)
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
        # Get the repulsion between not masked atoms and fixed atoms
        rep1,repulsion_deriv=self.energy_rest(None,i_nm,not_masked,False,cov_dis,distances,vec_distances,n_atoms,n_nm)
        # Get the repulsion between not masked atoms and not masked atoms
        rep2,repulsion_deriv=self.energy_nm_nm(None,i_nm,not_masked,False,cov_dis,distances,vec_distances,n_atoms,n_nm)
        return (self.r_scale**self.power)*(rep1+rep2)
        
    def get_energy_forces(self,atoms,**kwargs):
        " Get the energy and forces. "
        # Get the not fixed (not masked) atom indicies
        not_masked=np.array(self.get_constraints(atoms))
        # Get the number of atoms and the number of the fixed atoms
        n_atoms=len(atoms)
        n_nm=len(not_masked)
        i_nm=np.arange(n_nm).reshape(-1,1)
        repulsion_deriv=np.zeros((n_nm,3))
        # Get the covariance distance and the distances between atoms
        cov_dis,distances,vec_distances=self.get_cov_dis(atoms,not_masked,mic=self.mic,use_derivatives=True,i_nm=i_nm.flatten())
        # Get the repulsion between not masked atoms and fixed atoms
        rep1,repulsion_deriv=self.energy_rest(repulsion_deriv,i_nm,not_masked,True,cov_dis,distances,vec_distances,n_atoms,n_nm)
        # Get the repulsion between not masked atoms and not masked atoms
        rep2,repulsion_deriv=self.energy_nm_nm(repulsion_deriv,i_nm,not_masked,True,cov_dis,distances,vec_distances,n_atoms,n_nm)
        # Make the derivatives into forces
        deriv=np.zeros((n_atoms,3))
        deriv[not_masked]=-self.power*(self.r_scale**self.power)*repulsion_deriv
        return (self.r_scale**self.power)*(rep1+rep2),deriv.reshape(n_atoms,3)
    
    def energy_rest(self,repulsion_deriv,i_nm,not_masked,use_derivatives,cov_dis,distances,vec_distances,n_atoms,n_nm):
        " Get the energy contribution and their derivatives of non-fixed and fixed atoms. "
        if n_atoms!=n_nm:
            i_rest=np.setdiff1d(np.arange(n_atoms),not_masked)
            rep1=(cov_dis[i_nm,i_rest]/distances[i_nm,i_rest])**self.power
            if use_derivatives:
                repulsion_deriv+=np.sum(vec_distances[i_nm,i_rest]*((rep1/(distances[i_nm,i_rest]**2)).reshape(n_nm,n_atoms-n_nm,1)),axis=1)
            return np.sum(rep1),repulsion_deriv
        return 0.0,repulsion_deriv
    
    def energy_nm_nm(self,repulsion_deriv,i_nm,not_masked,use_derivatives,cov_dis,distances,vec_distances,n_atoms,n_nm):
        " Get the fingerprints and their derivatives of non-fixed and non-fixed atoms. "
        rep2=(cov_dis[i_nm,not_masked]/distances[i_nm,not_masked])**self.power
        if use_derivatives:
            repulsion_deriv+=np.sum(vec_distances[i_nm,not_masked]*((rep2/(distances[i_nm,not_masked]**2)).reshape(n_nm,n_nm,1)),axis=1)
        return 0.5*np.sum(rep2),repulsion_deriv
    
    def get_cov_dis(self,atoms,not_masked,mic,use_derivatives,i_nm,**kwargs):
        " Calculate the distances and scale them with the covalent radii. "
        # Get the distances and distance vectors
        distances,vec_distances=get_all_distances(atoms,not_masked,mic=mic,vector=use_derivatives)
        # Set the self distance to 1 to avoid errors
        distances[i_nm,not_masked]=1.0
        # Get all sums of covalent radii
        cov_dis=covalent_radii[atoms.get_atomic_numbers()]
        cov_dis=cov_dis[not_masked].reshape(-1,1)+cov_dis
        # Set the self distance to 0
        cov_dis[i_nm,not_masked]=0.0
        return cov_dis,distances,vec_distances
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(mic=self.mic,
                        reduce_dimensions=self.reduce_dimensions,
                        r_scale=self.r_scale,
                        power=self.power)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    