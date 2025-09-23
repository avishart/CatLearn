from numpy import asarray, identity
from .fingerprint import Fingerprint
from .geometry import get_constraints


class Cartesian(Fingerprint):
    """
    Fingerprint constructor class that convert an atoms instance into
    a fingerprint instance with vector and derivatives.
    The cartesian coordinate fingerprint is generated.
    """

    def make_fingerprint(self, atoms, **kwargs):
        "The calculation of the cartesian coordinates fingerprint"
        # Get the masked and not masked atoms
        not_masked, _ = get_constraints(
            atoms,
            reduce_dimensions=self.reduce_dimensions,
        )
        # Get the cartesian coordinates of the moved atoms
        vector = asarray(
            atoms.get_positions()[not_masked],
            dtype=self.dtype,
        ).reshape(-1)
        # Get the derivatives if requested
        if self.use_derivatives:
            derivative = identity(len(vector))
        else:
            derivative = None
        return vector, derivative
