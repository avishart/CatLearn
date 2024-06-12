import numpy as np
from .invdistances import InvDistances

class SortedDistances(InvDistances):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,periodic_softmax=True,mic=False,wrap=True,eps=1e-16,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        The sorted inverse distance fingerprint constructer class. 
        The inverse distances are scaled with covalent radii.

        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            periodic_softmax : bool
                Use a softmax weighting of the squared distances when periodic boundary conditions are used.
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
                         periodic_softmax=periodic_softmax,
                         mic=mic,
                         wrap=wrap,
                         eps=eps,
                         **kwargs)
                         
    def make_fingerprint(self,atoms,not_masked,masked,**kwargs):
        " Calculate the fingerprint and its derivative. "
        # Set parameters of array sizes
        n_atoms=len(atoms)
        n_nmasked=len(not_masked)
        n_masked=n_atoms-n_nmasked
        n_nm_m=n_nmasked*n_masked
        n_nm_nm=int(0.5*n_nmasked*(n_nmasked-1))
        n_total=n_nm_m+n_nm_nm
        # Make indicies arrays
        not_masked=np.array(not_masked,dtype=int)
        masked=np.array(masked,dtype=int)
        indicies=np.arange(n_atoms)
        i_nm=np.arange(n_nmasked)
        i_m=np.arange(n_masked)
        # Calculate all the fingerprints and their derivatives
        fij,gij,nmi,nmj=self.get_contributions(atoms,not_masked,masked,i_nm,n_total,n_nmasked,n_masked,n_nm_m)
        # Get all the indicies of the interactions
        indicies_nm_m,indicies_nm_nm=self.get_indicies(n_nmasked,n_masked,n_total,n_nm_m,nmi,nmj)
        # Make the arrays of fingerprints and their derivatives
        f=np.zeros(n_total)
        g=np.zeros((n_total,int(n_nmasked*3)))
        # Get all informations of the atoms and split them into types
        nmasked_indicies,masked_indicies,n_unique=self.element_setup(atoms,indicies,not_masked,masked,i_nm,i_m,nm_bool=True)
        # Get all combinations of the atom types
        combinations=zip(*np.triu_indices(n_unique,k=0,m=None))
        temp_len=0
        # Run over all combinations
        for ci,cj in combinations:
            # Find the indicies in the fingerprints for the combinations
            indicies_comb,len_i_comb=self.get_indicies_combination(ci,cj,nmasked_indicies,masked_indicies,indicies_nm_m,indicies_nm_nm)
            if len_i_comb:
                # Sort the fingerprints for the combinations
                len_new=temp_len+len_i_comb
                f,g=self.sort_fp(f,g,fij,gij,indicies_comb,temp_len,len_new)
                temp_len=len_new
        return f,g
    
    def sort_fp(self,f,g,fij,gij,indicies_comb,temp_len,len_new,**kwargs):
        " Sort the fingerprints after inverse distance magnitude. "
        i_sort=np.argsort(fij[indicies_comb])[::-1]
        i_sort=np.array(indicies_comb)[i_sort]
        f[temp_len:len_new]=fij[i_sort]
        if self.use_derivatives:
            g[temp_len:len_new]=gij[i_sort]
        return f,g
