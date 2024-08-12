from .invdistances import InvDistances


class InvDistances2(InvDistances):
    def __init__(
        self,
        reduce_dimensions=True,
        use_derivatives=True,
        periodic_softmax=True,
        mic=False,
        wrap=True,
        eps=1e-16,
        **kwargs,
    ):
        """
        Fingerprint constructer class that convert atoms object into
        a fingerprint object with vector and derivatives.
        The inverse squared distance fingerprint constructer class.
        The inverse squared distances are scaled with covalent radii.

        Parameters:
            reduce_dimensions : bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Calculate and store derivatives of the fingerprint wrt.
                the cartesian coordinates.
            periodic_softmax : bool
                Use a softmax weighting of the squared distances
                when periodic boundary conditions are used.
            mic : bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
                Either use mic or periodic_softmax, not both.
                mic is faster than periodic_softmax,
                but the derivatives are discontinuous.
            wrap: bool
                Whether to wrap the atoms to the unit cell or not.
            eps : float
                Small number to avoid division by zero.
        """
        # Set the arguments
        super().__init__(
            reduce_dimensions=reduce_dimensions,
            use_derivatives=use_derivatives,
            periodic_softmax=periodic_softmax,
            mic=mic,
            wrap=wrap,
            eps=eps,
            **kwargs,
        )

    def get_contributions(
        self,
        atoms,
        not_masked,
        masked,
        i_nm,
        n_total,
        n_nmasked,
        n_masked,
        n_nm_m,
        **kwargs,
    ):
        # Get the fingerprint and indicies from InvDistances
        f, g, nmi, nmj = super().get_contributions(
            atoms,
            not_masked,
            masked,
            i_nm,
            n_total,
            n_nmasked,
            n_masked,
            n_nm_m,
            **kwargs,
        )
        # Adjust the fingerprint so it is squared
        if self.use_derivatives:
            g = (2.0 * f).reshape(-1, 1) * g
        f = f**2
        return f, g, nmi, nmj
