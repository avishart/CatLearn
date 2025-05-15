from numpy import asarray, identity
from .fingerprint import Fingerprint
from .geometry import get_constraints


class Cartesian(Fingerprint):
    def __init__(
        self,
        reduce_dimensions=True,
        use_derivatives=True,
        dtype=None,
        **kwargs,
    ):
        """
        Fingerprint constructer class that convert atoms object into
        a fingerprint object with vector and derivatives.
        The cartesian coordinate fingerprint is generated.

        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt.
                the cartesian coordinates.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set the arguments
        super().__init__(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            dtype=dtype,
            **kwargs,
        )

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
