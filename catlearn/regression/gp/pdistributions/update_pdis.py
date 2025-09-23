from ..hpboundary.strict import StrictBoundaries


def update_pdis(
    model,
    parameters,
    X,
    Y,
    bounds=None,
    pdis=None,
    dtype=float,
    **kwargs,
):
    """
    Update given prior distribution of hyperparameters from
    educated guesses in log space.
    """
    # Check if prior distributions are given
    if pdis is None:
        return pdis
    # Make boundary conditions for updating the prior distributions
    if bounds is None:
        # Use strict educated guesses for the boundary conditions if not given
        bounds = StrictBoundaries(log=True, use_prior_mean=True, dtype=dtype)
        # Update boundary conditions to the data
        bounds.update_bounds(model, X, Y, parameters)
    # Make prior distributions for hyperparameters from boundary conditions
    for para, bound in bounds.get_bounds().items():
        if para in pdis.keys():
            # Use given prior distribution to update it
            pdis[para] = pdis[para].min_max(bound[:, 0], bound[:, 1])
    return pdis
