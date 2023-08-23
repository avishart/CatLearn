import numpy as np
from .meandistances import Mean_distances

class Mean_distances_power(Mean_distances):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,mic=True,power=2,use_roots=True,**kwargs):
        """ The Mean of inverse distance fingerprint scaled with covalent radii in different powers.
            Parameters:
                reduce_dimensions: bool
                    Whether to reduce the fingerprint space if constrains are used.
                use_derivatives: bool
                    Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
                mic: bool
                    Minimum Image Convention (Shortest distances when periodic boundary is used).
                power: int
                    The power of the inverse distances.
                use_roots: bool
                    Whether to use roots of the power elements.
        """
        super().__init__(reduce_dimensions=reduce_dimensions,use_derivatives=use_derivatives,mic=mic,**kwargs)
        self.power=power
        self.use_roots=use_roots
    
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
                # Mean the fingerprints for the combinations
                f,g=self.mean_fp_power(f,g,fij,gij,indicies_comb,len_i_comb)
        return np.array(f),np.array(g)
    
    def mean_fp_power(self,f,g,fij,gij,indicies_comb,len_i_comb,**kwargs):
        " Mean of the fingerprints. "
        powers=np.arange(1,self.power+1)
        fij_powers=fij[indicies_comb].reshape(-1,1)**powers
        fij_means=np.mean(fij_powers,axis=0)
        if self.use_roots:
            f.extend(fij_means**(1.0/powers))
        else:
            f.extend(fij_means)
        if self.use_derivatives:
            g.append(np.mean(gij[indicies_comb],axis=0))
            fg_prod=np.mean(fij_powers[:,:-1].T.reshape(self.power-1,len_i_comb,1)*gij[indicies_comb],axis=1)
            if self.use_roots:
                g.extend(fg_prod*(fij_means[1:]**((1.0-powers[1:])/powers[1:])).reshape(-1,1))
            else:
                g.extend(powers[1:].reshape(-1,1)*fg_prod)
        return f,g
            
    def copy(self):
        " Copy the Fingerprint. "
        return self.__class__(reduce_dimensions=self.reduce_dimensions,use_derivatives=self.use_derivatives,mic=self.mic,power=self.power,use_roots=self.use_roots)
    
    def __repr__(self):
        return "Mean_distances_power(reduce_dimensions={},use_derivatives={},mic={},power={},use_roots={})".format(self.reduce_dimensions,self.use_derivatives,self.mic,self.power,self.use_roots)