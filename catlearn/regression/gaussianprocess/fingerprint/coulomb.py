import numpy as np
from .invdistances import InvDistances
from .geometry import get_all_distances

class Coulomb(InvDistances):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,mic=False,wrap=True,eps=1e-16,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        The Coulomb matrix fingerprint is generated.

        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary conditions are used).
                Either use mic or periodic_softmax, not both. mic is faster than periodic_softmax, but the derivatives are discontinuous.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            eps : float
                Small number to avoid division by zero.
        """
        # Set the arguments
        super().__init__(reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         mic=mic,
                         wrap=wrap,
                         eps=eps,
                         **kwargs)
    
    def make_fingerprint(self,atoms,not_masked,masked,**kwargs):
        " The calculation of the coulomb matrix fingerprint "
        vector,derivative=self.calculate_coulomb(atoms,not_masked,use_derivatives=self.use_derivatives,mic=self.mic)
        return vector,derivative
    
    def calculate_coulomb(self,atoms,not_masked,use_derivatives=True,mic=True):
        " The actually calculation of the coulomb matrix fingerprint "
        cmatrix,distances,vec_distances=self.get_coulomb(atoms,use_derivatives=use_derivatives,mic=mic)
        i_sort=np.argsort(np.linalg.norm(cmatrix,axis=0))[::-1]
        i_triu=list(zip(*np.triu_indices(len(cmatrix))))
        cmatrix=(cmatrix[i_sort,:])[:,i_sort].copy()
        if use_derivatives:
            distances=(distances[i_sort,:])[:,i_sort].copy()
            vec_distances=(vec_distances[i_sort,:])[:,i_sort].copy()
            n_atoms_g=len(not_masked)
            g=self.fp_deriv_iter(i_triu,cmatrix,distances,vec_distances,n_atoms_g,not_masked)
        else:
            cmatrix=cmatrix[i_sort,:].copy()
            cmatrix=cmatrix[:,i_sort].copy()
            g=None
        fp=np.array([cmatrix[i,j] for i,j in i_triu])
        return fp,g
    
    def get_coulomb(self,atoms,use_derivatives=False,mic=True,wrap=True):
        " Get distances and charges to calculate coulomb potential "
        from scipy.spatial.distance import squareform
        range_atoms=np.arange(len(atoms))
        distances,vec_distances=get_all_distances(atoms,range_atoms,mic=mic,vector=use_derivatives,wrap=wrap)
        atom_numb=np.array([float(an) for an in atoms.get_atomic_numbers()])
        atom_numb=atom_numb.reshape(-1,1)*atom_numb
        atom_numb[range_atoms,range_atoms]=0.0
        atom_numb=squareform(atom_numb)
        cmatrix=squareform(atom_numb/squareform(distances))
        cmatrix[range_atoms,range_atoms]=0.5*(atoms.get_atomic_numbers()**2.4)
        if use_derivatives:
            return cmatrix,distances,vec_distances
        return cmatrix,distances,None
    
    def fp_deriv_iter(self,i_triu,cmatrix,distances,vec_distances,n_atoms_g,not_masked):
        " Calculate the derivative of the coulomb matrix wrt to cartesian coordinates "
        g=[]
        for elei,elej in i_triu:
            gij=[0.0]*(n_atoms_g*3)
            if elei!=elej:
                if elei in not_masked or elej in not_masked:                    
                    gij_value=(cmatrix[elei,elej]/(distances[elei,elej]**2))*vec_distances[elei,elej]
                    if elei in not_masked:
                        i=not_masked.index(elei)
                        gij[3*i:3*i+3]=gij_value
                    if elej in not_masked:
                        j=not_masked.index(elej)
                        gij[3*j:3*j+3]=-gij_value
            g.append(gij)
        return np.array(g)
    