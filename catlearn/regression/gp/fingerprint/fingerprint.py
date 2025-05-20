from numpy import array, finfo
from .geometry import get_constraints
from .fingerprintobject import FingerprintObject


class Fingerprint:
    """
    Fingerprint constructor class that convert an atoms instance into
    a fingerprint instance with vector and derivatives.
    """

    def __init__(
        self,
        reduce_dimensions=True,
        use_derivatives=True,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the fingerprint constructor.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Calculate and store derivatives of the fingerprint wrt.
                the cartesian coordinates.
            dtype: type (optional)
                The data type of the arrays.
                If None, the default data type is used.
        """
        # Set the arguments
        self.update_arguments(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            dtype=dtype,
            **kwargs,
        )

    def __call__(self, atoms, **kwargs):
        """
        Convert atoms to fingerprint and return the fingerprint object.

        Parameters:
            atoms: ASE Atoms
                The ASE Atoms object that are converted to a fingerprint.

        Returns:
            FingerprintObject: Object with the fingerprint array and
            its derivatives if requested.
        """
        # Calculate the fingerprint and its derivatives if requested
        vector, derivative = self.make_fingerprint(
            atoms,
            **kwargs,
        )
        # Make the fingerprint object and store the arrays within
        if self.use_derivatives:
            return FingerprintObject(vector=vector, derivative=derivative)
        return FingerprintObject(vector=vector, derivative=None)

    def get_use_derivatives(self):
        "Get whether the derivatives of the targets are used."
        return self.use_derivatives

    def get_reduce_dimensions(self):
        """
        Get whether the reduction of the fingerprint space is used
        if constrains are used.
        """
        return self.reduce_dimensions

    def set_use_derivatives(self, use_derivatives, **kwargs):
        """
        Set whether to use derivatives/forces in the targets.

        Parameters:
            use_derivatives: bool
                Whether to use derivatives/forces in the targets.

        Returns:
            self: The updated object itself.
        """
        # Set the use derivatives
        self.use_derivatives = use_derivatives
        return self

    def set_reduce_dimensions(self, reduce_dimensions, **kwargs):
        """
        Set whether to reduce the fingerprint space if constrains are used.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.

        Returns:
            self: The updated object itself.
        """
        # Set the reduce dimensions
        self.reduce_dimensions = reduce_dimensions
        return self

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the data type
        self.dtype = dtype
        # Set a small number to avoid division by zero
        self.eps = 1.1 * finfo(self.dtype).eps
        return self

    def update_arguments(
        self,
        reduce_dimensions=None,
        use_derivatives=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the class with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives: bool
                Calculate and store derivatives of the fingerprint wrt.
                the cartesian coordinates.
            dtype: type
                The data type of the arrays.
                If None, the default data type is used.

        Returns:
            self: The updated instance itself.
        """
        if reduce_dimensions is not None:
            self.set_reduce_dimensions(reduce_dimensions)
        if use_derivatives is not None:
            self.set_use_derivatives(use_derivatives)
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype=dtype)
        if not hasattr(self, "not_masked"):
            self.not_masked = None
        if not hasattr(self, "masked"):
            self.masked = None
        return self

    def make_fingerprint(self, atoms, **kwargs):
        "The calculation of the fingerprint"
        raise NotImplementedError()

    def get_not_masked(self, atoms, masked=None, recalc=False, **kwargs):
        "Get the not masked atoms."
        # Use the stored values if recalculation is not requested
        if not recalc and self.not_masked is not None:
            return self.not_masked
        # Recalculate the not masked atoms
        if masked is None:
            not_masked, masked = get_constraints(
                atoms,
                reduce_dimensions=self.reduce_dimensions,
                **kwargs,
            )
            self.masked = array(masked, dtype=int)
        else:
            i_all = set(range(len(atoms)))
            not_masked = list(i_all.difference(set(masked)))
            not_masked = sorted(not_masked)
        self.not_masked = array(not_masked, dtype=int)
        return self.not_masked

    def get_masked(self, atoms, not_masked=None, recalc=False, **kwargs):
        "Get the masked atoms."
        # Use the stored values if recalculation is not requested
        if not recalc and self.masked is not None:
            return self.masked
        if not_masked is None:
            not_masked, masked = get_constraints(
                atoms,
                reduce_dimensions=self.reduce_dimensions,
                **kwargs,
            )
            self.not_masked = array(not_masked, dtype=int)
        else:
            i_all = set(range(len(atoms)))
            masked = list(i_all.difference(set(not_masked)))
            masked = sorted(masked)
        self.masked = array(masked, dtype=int)
        return self.masked

    def reset_masked(self):
        "Reset the masked atoms."
        self.masked = None
        self.not_masked = None
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            reduce_dimensions=self.reduce_dimensions,
            use_derivatives=self.use_derivatives,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)
