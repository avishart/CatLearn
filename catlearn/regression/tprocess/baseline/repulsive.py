import numpy as np
from ase.calculators.calculator import Calculator
from .baseline import Baseline_calculator
        
class Repulsion_calculator(Baseline_calculator):
    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True
    
    def __init__(self,r_scale=0.7):
        """ A baseline calculator for ASE atoms object. 
        It uses a repulsive Lennard-Jones potential baseline.    
        """
        Calculator.__init__(self)
        self.r_scale=r_scale
        pass
        
    def get_energy(self,atoms):
        " Get only the energy. "
        from ase.data import covalent_radii
        n_atoms=len(atoms)
        distances2=atoms.get_all_distances(mic=True,vector=False)
        distances2[range(n_atoms),range(n_atoms)]=1.0
        cov_dis=covalent_radii[atoms.get_atomic_numbers()]
        cov_dis=(cov_dis.reshape(-1,1)+cov_dis)
        cov_dis[range(n_atoms),range(n_atoms)]=0.0
        repulsion_matrix=(cov_dis/distances2)**12
        return 0.5*(self.r_scale**12)*np.sum(repulsion_matrix)
        
    def get_energy_forces(self,atoms):
        " Get the energy and forces. "
        from ase.data import covalent_radii
        n_atoms=len(atoms)
        dis_matrix=atoms.get_all_distances(mic=True,vector=True)
        distances2=np.sum(dis_matrix**2,axis=2)
        distances2[range(n_atoms),range(n_atoms)]=1.0
        cov_dis=covalent_radii[atoms.get_atomic_numbers()]
        cov_dis=(cov_dis.reshape(-1,1)+cov_dis)**2
        cov_dis[range(n_atoms),range(n_atoms)]=0.0
        repulsion_matrix=(cov_dis/distances2)**6
        repulsion_deriv=dis_matrix*((repulsion_matrix/distances2).reshape(n_atoms,n_atoms,1))
        return 0.5*(self.r_scale**12)*np.sum(repulsion_matrix),6*(self.r_scale**12)*np.sum(repulsion_deriv,axis=0)
    
