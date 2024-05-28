import numpy as np
from .baseline import BaselineCalculator
from ..fingerprint.geometry import get_inverse_distances
        
class RepulsionCalculator(BaselineCalculator):
    implemented_properties=['energy', 'forces']
    nolabel=True
    
    def __init__(self,reduce_dimensions=True,r_scale=0.7,power=12,periodic_softmax=True,mic=False,wrap=True,eps=1e-16,**kwargs):
        """ 
        A baseline calculator for ASE atoms object. 
        It uses a repulsive Lennard-Jones potential baseline.  
        The power and the scaling of the repulsive Lennard-Jones potential can be selected.

        Parameters: 
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
            r_scale : float
                The scaling of the covalent radii. 
                A smaller value will move the repulsion to a lower distances. 
            power : int
                The power of the repulsion.
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
        super().__init__(reduce_dimensions=reduce_dimensions,
                         r_scale=r_scale,
                         power=power,
                         periodic_softmax=periodic_softmax,
                         mic=mic,
                         wrap=wrap,
                         eps=eps,
                         **kwargs)

    def update_arguments(self,reduce_dimensions=None,r_scale=None,power=None,periodic_softmax=None,mic=None,wrap=None,eps=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters: 
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms if constrains are used.
            r_scale : float
                The scaling of the covalent radii. 
                A smaller value will move the repulsion to a lower distances. 
            power : int
                The power of the repulsion.
            periodic_softmax : bool
                Use a softmax weighting of the squared distances when periodic boundary conditions are used.
            mic : bool
                Minimum Image Convention (Shortest distances when periodic boundary conditions are used).
                Either use mic or periodic_softmax, not both. mic is faster than periodic_softmax, but the derivatives are discontinuous.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            eps : float
                Small number to avoid division by zero.
                
        Returns:
            self: The updated object itself.
        """
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        if r_scale is not None:
            self.r_scale=r_scale
        if power is not None:
            self.power=int(power)
        if periodic_softmax is not None:
            self.periodic_softmax=periodic_softmax
        if mic is not None:
            self.mic=mic
        if wrap is not None:
            self.wrap=wrap
        if eps is not None:
            self.eps=abs(float(eps))
        # Calculate the normalization
        self.c0=self.r_scale**self.power
        return self
    
    def get_energy_forces(self,atoms,get_derivatives=True,**kwargs):
        " Get the energy and forces. "
        # Get the not fixed (not masked) atom indicies
        not_masked,masked=self.get_constraints(atoms)
        not_masked=np.array(not_masked)
        masked=np.array(masked)
        # Get the inverse distances
        f,g=self.get_inv_distances(atoms,not_masked,masked,get_derivatives,**kwargs)
        # Calculate energy
        energy=self.c0*np.sum(f**self.power)
        if get_derivatives:
            forces=np.zeros((len(atoms),3))
            derivs=np.sum(((-self.c0*self.power)*(f**(self.power-1))).reshape(-1,1)*g,axis=0)
            forces[not_masked]=derivs.reshape(-1,3)
            return energy,forces
        return energy

    def get_inv_distances(self,atoms,not_masked,masked,get_derivatives,**kwargs):
        " Get the unique inverse distances scaled with the covalent radii and its derivatives. "
        # Get the indicies for not fixed and not fixed atoms interactions
        nmi,nmj=np.triu_indices(len(not_masked),k=1,m=None)
        nmi_ind=not_masked[nmi]
        nmj_ind=not_masked[nmj]
        f,g=get_inverse_distances(atoms,not_masked=not_masked,masked=masked,nmi=nmi,nmj=nmj,nmi_ind=nmi_ind,nmj_ind=nmj_ind,use_derivatives=get_derivatives,use_covrad=True,periodic_softmax=self.periodic_softmax,mic=self.mic,wrap=self.wrap,eps=self.eps,**kwargs)
        return f,g
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(reduce_dimensions=self.reduce_dimensions,
                        r_scale=self.r_scale,
                        power=self.power,
                        periodic_softmax=self.periodic_softmax,
                        mic=self.mic,
                        wrap=self.wrap,
                        eps=self.eps)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    