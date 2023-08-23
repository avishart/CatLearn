import numpy as np
from .invdistances import Inv_distances

class Sum_distances(Inv_distances):
    " The sum of inverse distance fingerprint scaled with covalent radii. "
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " Calculate the fingerprint and its derivative. "
        # Set parameters of array sizes
        n_atoms=len(atoms)
        n_nmasked=len(not_masked)
        n_masked=n_atoms-n_nmasked
        n_nm_m=n_nmasked*n_masked
        n_nm_nm=int(0.5*n_nmasked*(n_nmasked-1))
        n_total=n_nm_m+n_nm_nm
        # Make indicies arrays
        not_masked=np.array(not_masked)
        indicies=np.arange(n_atoms)
        masked=np.setdiff1d(indicies,not_masked)
        i_nm=np.arange(n_nmasked)
        i_m=np.arange(n_masked)
        # Calculate all the fingerprints and their derivatives
        fij,gij,nmi,nmj=self.get_contributions(atoms,not_masked,masked,i_nm,n_total,n_nmasked,n_masked,n_nm_m)
        # Get all the indicies of the interactions
        indicies_nm_m,indicies_nm_nm=self.get_indicies(n_nmasked,n_masked,n_total,n_nm_m,nmi,nmj)
        # Make the arrays of fingerprints and their derivatives
        f=[]
        g=[]
        # Get all informations of the atoms and split them into types
        nmasked_indicies,masked_indicies,n_unique=self.element_setup(atoms,indicies,not_masked,masked,i_nm,i_m,nm_bool=True)
        # Get all combinations of the atom types
        combinations=zip(*np.triu_indices(n_unique,k=0,m=None))
        # Run over all combinations
        for ci,cj in combinations:
            # Find the indicies in the fingerprints for the combinations
            indicies_comb,len_i_comb=self.get_indicies_combination(ci,cj,nmasked_indicies,masked_indicies,indicies_nm_m,indicies_nm_nm)
            if len_i_comb:
                # Sum the fingerprints for the combinations
                f,g=self.sum_fp(f,g,fij,gij,indicies_comb)
        return np.array(f),np.array(g)
    
    def sum_fp(self,f,g,fij,gij,indicies_comb,**kwargs):
        " Sum of the fingerprints. "
        f.append(np.sum(fij[indicies_comb]))
        if self.use_derivatives:
            g.append(np.sum(gij[indicies_comb],axis=0))
        return f,g
            
    def copy(self):
        " Copy the Fingerprint. "
        return self.__class__(reduce_dimensions=self.reduce_dimensions,use_derivatives=self.use_derivatives,mic=self.mic)
    
    def __repr__(self):
        return "Sum_distances(reduce_dimensions={},use_derivatives={},mic={})".format(self.reduce_dimensions,self.use_derivatives,self.mic)
    