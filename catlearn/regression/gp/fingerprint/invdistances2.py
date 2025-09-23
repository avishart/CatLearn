from .invdistances import InvDistances


class InvDistances2(InvDistances):
    """
    Fingerprint constructor class that convert an atoms instance into
    a fingerprint instance with vector and derivatives.
    The squared inverse distances are constructed as the fingerprint.
    The squared inverse distances are scaled with covalent radii.
    """

    def modify_fp(
        self,
        fp,
        g,
        atomic_numbers,
        tags,
        not_masked,
        masked,
        nmi,
        nmj,
        nmi_ind,
        nmj_ind,
        use_include_ncells=False,
        **kwargs,
    ):
        "Modify the fingerprint."
        # Adjust the derivatives so they are squared
        if g is not None:
            g = (2.0 * fp)[..., None] * g
            g = self.insert_to_deriv_matrix(
                g=g,
                not_masked=not_masked,
                masked=masked,
                nmi=nmi,
                nmj=nmj,
                use_include_ncells=use_include_ncells,
            )
        # Reshape the fingerprint
        if use_include_ncells:
            fp = fp.reshape(-1)
        # Adjust the fingerprint so it is squared
        fp = fp**2
        return fp, g
