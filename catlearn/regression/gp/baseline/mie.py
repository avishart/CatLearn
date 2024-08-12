import numpy as np
from .repulsive import RepulsionCalculator


class MieCalculator(RepulsionCalculator):
    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(
        self,
        reduce_dimensions=True,
        r_scale=1.0,
        denergy=0.1,
        power_r=8,
        power_a=6,
        periodic_softmax=True,
        mic=False,
        wrap=True,
        eps=1e-16,
        **kwargs,
    ):
        """
        A baseline calculator for ASE atoms object.
        It uses the Mie potential baseline.
        The power and the scaling of the Mie potential can be selected.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms
                if constrains are used.
            r_scale : float
                The scaling of the covalent radii.
                A smaller value will move the potential to a lower distances.
            denergy : float
                The dispersion energy of the potential.
            power_r : int
                The power of the potential part.
            power_a : int
                The power of the attraction part.
            periodic_softmax : bool
                Use a softmax weighting of the squared distances
                when periodic boundary conditions are used.
            mic : bool
                Minimum Image Convention (Shortest distances
                when periodic boundary conditions are used).
                Either use mic or periodic_softmax, not both.
                mic is faster than periodic_softmax,
                but the derivatives are discontinuous.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            eps : float
                Small number to avoid division by zero.
        """
        super().__init__(
            reduce_dimensions=reduce_dimensions,
            r_scale=r_scale,
            denergy=denergy,
            power_a=power_a,
            power_r=power_r,
            periodic_softmax=periodic_softmax,
            mic=mic,
            wrap=wrap,
            eps=eps,
            **kwargs,
        )

    def update_arguments(
        self,
        reduce_dimensions=None,
        r_scale=None,
        denergy=None,
        power_r=None,
        power_a=None,
        periodic_softmax=None,
        mic=None,
        wrap=None,
        eps=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the dimensions to only moving atoms
                if constrains are used.
            r_scale : float
                The scaling of the covalent radii.
                A smaller value will move the potential to a lower distances.
            denergy : float
                The dispersion energy of the potential.
            power_r : int
                The power of the potential part.
            power_a : int
                The power of the attraction part.
            periodic_softmax : bool
                Use a softmax weighting of the squared distances
                when periodic boundary conditions are used.
            mic : bool
                Minimum Image Convention (Shortest distances
                when periodic boundary conditions are used).
                Either use mic or periodic_softmax, not both.
                mic is faster than periodic_softmax,
                but the derivatives are discontinuous.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            eps : float
                Small number to avoid division by zero.

        Returns:
            self: The updated object itself.
        """
        if reduce_dimensions is not None:
            self.reduce_dimensions = reduce_dimensions
        if r_scale is not None:
            self.r_scale = float(r_scale)
        if denergy is not None:
            self.denergy = float(denergy)
        if power_r is not None:
            self.power_r = int(power_r)
        if power_a is not None:
            self.power_a = int(power_a)
        if periodic_softmax is not None:
            self.periodic_softmax = periodic_softmax
        if mic is not None:
            self.mic = mic
        if wrap is not None:
            self.wrap = wrap
        if eps is not None:
            self.eps = abs(float(eps))
        # Calculate the normalization
        power_ar = self.power_a / (self.power_r - self.power_a)
        c0 = self.denergy * (
            ((self.power_r / self.power_a) ** power_ar)
            * (self.power_r / (self.power_r - self.power_a))
        )
        # Calculate the r_scale powers
        self.r_scale_r = c0 * (self.r_scale**self.power_r)
        self.r_scale_a = c0 * (self.r_scale**self.power_a)
        return self

    def get_energy_forces(self, atoms, get_derivatives=True, **kwargs):
        "Get the energy and forces."
        # Get the not fixed (not masked) atom indicies
        not_masked, masked = self.get_constraints(atoms)
        not_masked = np.array(not_masked, dtype=int)
        masked = np.array(masked, dtype=int)
        # Get the inverse distances
        f, g = self.get_inv_distances(
            atoms,
            not_masked,
            masked,
            get_derivatives,
            **kwargs,
        )
        # Calculate energy
        energy = (self.r_scale_r * np.sum(f**self.power_r)) - (
            self.r_scale_a * np.sum(f**self.power_a)
        )
        if get_derivatives:
            forces = np.zeros((len(atoms), 3))
            power_ar = self.power_a * self.r_scale_a
            power_rr = self.power_r * self.r_scale_r
            inner = (power_ar * (f ** (self.power_a - 1))) - (
                power_rr * (f ** (self.power_r - 1))
            )
            derivs = np.sum(inner.reshape(-1, 1) * g, axis=0)
            forces[not_masked] = derivs.reshape(-1, 3)
            return energy, forces
        return energy

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            reduce_dimensions=self.reduce_dimensions,
            r_scale=self.r_scale,
            denergy=self.denergy,
            power_a=self.power_a,
            power_r=self.power_r,
            periodic_softmax=self.periodic_softmax,
            mic=self.mic,
            wrap=self.wrap,
            eps=self.eps,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
