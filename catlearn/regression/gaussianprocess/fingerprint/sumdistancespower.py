import numpy as np
from .sumdistances import SumDistances

class SumDistancesPower(SumDistances):
    def __init__(self,reduce_dimensions=True,use_derivatives=True,periodic_softmax=True,mic=False,wrap=True,eps=1e-16,power=2,use_roots=True,**kwargs):
        """ 
        Fingerprint constructer class that convert atoms object into a fingerprint object with vector and derivatives.
        The sum of dfferent powers of the inverse distances fingerprint constructer class. 
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
            power: int
                The power of the inverse distances.
            use_roots: bool
                Whether to use roots of the power elements.
        """
        # Set the arguments
        super().__init__(reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         periodic_softmax=periodic_softmax,
                         mic=mic,
                         wrap=wrap,
                         eps=eps,
                         power=power,
                         use_roots=use_roots,
                         **kwargs)
        
    def update_arguments(self,reduce_dimensions=None,use_derivatives=None,periodic_softmax=None,mic=None,wrap=None,eps=None,power=None,use_roots=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

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
            power: int
                The power of the inverse distances.
            use_roots: bool
                Whether to use roots of the power elements.

        Returns:
            self: The updated instance itself.
        """
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        if periodic_softmax is not None:
            self.periodic_softmax=periodic_softmax
        if mic is not None:
            self.mic=mic
        if wrap is not None:
            self.wrap=wrap
        if eps is not None:
            self.eps=abs(float(eps))
        if power is not None:
            self.power=int(power)
        if use_roots is not None:
            self.use_roots=use_roots
        return self
    
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
                f,g=self.sum_fp_power(f,g,fij,gij,indicies_comb,len_i_comb)
        return np.array(f),np.array(g)
    
    def sum_fp_power(self,f,g,fij,gij,indicies_comb,len_i_comb,**kwargs):
        " Sum of the fingerprints. "
        powers=np.arange(1,self.power+1)
        fij_powers=fij[indicies_comb].reshape(-1,1)**powers
        fij_sums=np.sum(fij_powers,axis=0)
        if self.use_roots:
            f.extend(fij_sums**(1.0/powers))
        else:
            f.extend(fij_sums)
        if self.use_derivatives:
            g.append(np.sum(gij[indicies_comb],axis=0))
            fg_prod=np.sum(fij_powers[:,:-1].T.reshape(self.power-1,len_i_comb,1)*gij[indicies_comb],axis=1)
            if self.use_roots:
                g.extend(fg_prod*(fij_sums[1:]**((1.0-powers[1:])/powers[1:])).reshape(-1,1))
            else:
                g.extend(powers[1:].reshape(-1,1)*fg_prod)
        return f,g
            
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(reduce_dimensions=self.reduce_dimensions,
                        use_derivatives=self.use_derivatives,
                        periodic_softmax=self.periodic_softmax,
                        mic=self.mic,
                        wrap=self.wrap,
                        eps=self.eps,
                        power=self.power,
                        use_roots=self.use_roots)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    