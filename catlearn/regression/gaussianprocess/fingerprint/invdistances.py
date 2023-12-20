import numpy as np
import itertools
from ase.data import covalent_radii
from .fingerprint import Fingerprint
from .geometry import get_all_distances

class Inv_distances(Fingerprint):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,mic=True,sorting=True,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        The inverse distance fingerprint constructer class where the sizes are sorted and scaled with covalent radii. 
        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            sorting: bool
                Whether to sort the inverse distances after size or not.
        """
        # Set the arguments
        super().__init__(reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         mic=mic,
                         sorting=sorting,
                         **kwargs)
        
    def update_arguments(self,reduce_dimensions=None,use_derivatives=None,mic=None,sorting=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt. the cartesian coordinates.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary is used).
            sorting: bool
                Whether to sort the inverse distances after size or not.
        Returns:
            self: The updated object itself.
        """
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        if mic is not None:
            self.mic=mic
        if sorting is not None:
            self.sorting=sorting
        return self
                         
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
        # Return the fingerprints and their derivatives it is not sorted
        if not self.sorting:
            return fij,gij
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
    
    def element_setup(self,atoms,indicies,not_masked=None,masked=None,i_nm=None,i_m=None,nm_bool=True,**kwargs):
        " Get all informations of the atoms and split them into types. "
        # Merge element type and their tags
        combis=list(zip(atoms.get_atomic_numbers(),atoms.get_tags()))
        # Find all unique combinations
        unique_combis=np.array(list(set(combis)))
        n_unique=len(unique_combis)
        # Get the Booleans for what combination it belongs to
        bools=np.all(np.array(combis).reshape(-1,1,2)==unique_combis,axis=2)
        if not nm_bool:
            split_indicies=[indicies[ind] for ind in bools.T]
            return split_indicies,split_indicies.copy(),n_unique
        # Classify all non-fixed atoms in their unique combination
        nmasked_indicies=[i_nm[ind] for ind in bools[not_masked].T]
        # Classify all fixed atoms in their unique combination
        masked_indicies=[i_m[ind] for ind in bools[masked].T]
        return nmasked_indicies,masked_indicies,n_unique
    
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
    
    def get_contributions(self,atoms,not_masked,masked,i_nm,n_total,n_nmasked,n_masked,n_nm_m,**kwargs):
        " Calculate all the fingerprints and their derivatives. "
        # Calculate the covariance radii, distances, and distance vectors 
        cov_dis,distances,vec_distances=self.get_cov_dis(atoms,not_masked,self.mic,self.use_derivatives,i_nm)
        # Make the arrays of fingerprints and their derivatives
        fij=np.zeros(n_total)
        gij=np.zeros((n_total,int(n_nmasked*3)))
        # Calculate the fingerprints of not fixed (not masked) and fixed atoms
        if n_nm_m:
            i_nm_re=i_nm.reshape(-1,1)
            fij[:n_nm_m]=(cov_dis[i_nm_re,masked]/distances[i_nm_re,masked]).reshape(-1)
            # Calculate the derivatives of not fixed (not masked) and fixed atoms
            if self.use_derivatives:
                i_g=np.repeat(np.arange(n_nm_m),3)
                j_g=np.tile(3*i_nm_re+np.array([0,1,2]),(1,n_masked)).reshape(-1)
                gij[i_g,j_g]=(vec_distances[i_nm_re,masked]*((cov_dis[i_nm_re,masked]/(distances[i_nm_re,masked]**3)).reshape(n_nmasked,n_masked,1))).reshape(-1)
        # Get the indicies for not fixed and not fixed atoms interactions
        nmi,nmj=np.triu_indices(n_nmasked,k=1,m=None)
        if n_total-n_nm_m:
            nmj_ind=not_masked[nmj]
            # Calculate the fingerprints of not fixed and not fixed atoms
            fij[n_nm_m:]=cov_dis[nmi,nmj_ind]/distances[nmi,nmj_ind]
            if self.use_derivatives:
                # Calculate the derivatives of not fixed and not fixed atoms
                i_g=np.repeat(np.arange(n_nm_m,n_total),3)
                j_gi=(3*nmi.reshape(-1,1)+np.array([0,1,2])).reshape(-1)
                j_gj=(3*nmj.reshape(-1,1)+np.array([0,1,2])).reshape(-1)
                gij[i_g,j_gi]=(vec_distances[nmi,nmj_ind]*(cov_dis[nmi,nmj_ind]/(distances[nmi,nmj_ind]**3)).reshape(-1,1)).reshape(-1)
                gij[i_g,j_gj]=-gij[i_g,j_gi]
        return fij,gij,nmi,nmj
    
    def get_indicies(self,n_nmasked,n_masked,n_total,n_nm_m,nmi,nmj,**kwargs):
        " Get all the indicies of the interactions. "
        # Make the indicies of not fixed (not masked) and fixed atoms interactions
        indicies_nm_m=np.arange(n_nm_m,dtype=int).reshape(n_nmasked,n_masked)
        # Make the indicies of not fixed and fixed atoms interactions
        indicies_nm_nm=np.zeros((n_nmasked,n_nmasked),dtype=int)
        indicies_nm_nm[nmi,nmj]=indicies_nm_nm[nmj,nmi]=np.arange(n_nm_m,n_total,dtype=int)
        return indicies_nm_m,indicies_nm_nm
    
    def get_indicies_combination(self,ci,cj,nmasked_indicies,masked_indicies,indicies_nm_m,indicies_nm_nm,**kwargs):
        " Get all the indicies in the fingerprint for the specific combination of atom types. "
        indicies_comb=[]
        i_nm_ci=nmasked_indicies[ci].reshape(-1,1)
        if ci==cj:
            indicies_comb=list(indicies_nm_m[i_nm_ci,masked_indicies[cj]].reshape(-1))
            ind_prod=np.array(list(itertools.combinations(nmasked_indicies[ci],2)))
            if len(ind_prod):
                indicies_comb=indicies_comb+list(indicies_nm_nm[ind_prod[:,0],ind_prod[:,1]])
        else:
            indicies_comb=list(indicies_nm_m[i_nm_ci,masked_indicies[cj]].reshape(-1))
            indicies_comb=indicies_comb+list(indicies_nm_m[nmasked_indicies[cj].reshape(-1,1),masked_indicies[ci]].reshape(-1))
            indicies_comb=indicies_comb+list(indicies_nm_nm[i_nm_ci,nmasked_indicies[cj]].reshape(-1))
        return indicies_comb,len(indicies_comb)
            
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(reduce_dimensions=self.reduce_dimensions,
                        use_derivatives=self.use_derivatives,
                        mic=self.mic,
                        sorting=self.sorting)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
