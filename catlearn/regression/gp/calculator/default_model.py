import warnings


def get_default_model(
    model="tp",
    prior="median",
    use_derivatives=True,
    use_fingerprint=False,
    global_optimization=True,
    parallel=False,
    n_reduced=None,
    round_hp=3,
    seed=None,
    dtype=float,
    model_kwargs={},
    prior_kwargs={},
    kernel_kwargs={},
    hpfitter_kwargs={},
    optimizer_kwargs={},
    lineoptimizer_kwargs={},
    function_kwargs={},
    **kwargs,
):
    """
    Get the default ML model from the simple given arguments.

    Parameters:
        model: str
            Either the tp that gives the Studen T process or
            gp that gives the Gaussian process.
        prior: str
            Specify what prior mean should be used.
        use_derivatives: bool
            Whether to use derivatives of the targets.
        use_fingerprint: bool
            Whether to use fingerprints for the features.
            This has to be the same as for the database!
        global_optimization: bool
            Whether to perform a global optimization of the hyperparameters.
            A local optimization is used if global_optimization=False,
            which can not be parallelized.
        parallel: bool
            Whether to optimize the hyperparameters in parallel.
        n_reduced: int or None
            If n_reduced is an integer, the hyperparameters are only optimized
            when the data set size is equal to or below the integer.
            If n_reduced is None, the hyperparameter is always optimized.
        round_hp: int (optional)
            The number of decimals to round the hyperparameters to.
            If None, the hyperparameters are not rounded.
        seed: int (optional)
            The random seed for the optimization.
            The seed an also be a RandomState or Generator instance.
            If not given, the default random number generator is used.
        dtype: type
            The data type of the arrays.
        model_kwargs: dict (optional)
            The keyword arguments for the model.
            The additional arguments are passed to the model.
        prior_kwargs: dict (optional)
            The keyword arguments for the prior mean.
        kernel_kwargs: dict (optional)
            The keyword arguments for the kernel.
        hpfitter_kwargs: dict (optional)
            The keyword arguments for the hyperparameter fitter.
        optimizer_kwargs: dict (optional)
            The keyword arguments for the optimizer.
        lineoptimizer_kwargs: dict (optional)
            The keyword arguments for the line optimizer.
        function_kwargs: dict (optional)
            The keyword arguments for the objective function.

    Returns:
        model: Model
            The Machine Learning Model with kernel and
            prior that are optimized.
    """
    # Check that the model is given as a string
    if not isinstance(model, str):
        return model
    # Make the prior mean from given string
    if isinstance(prior, str):
        from ..means import Prior_median, Prior_mean, Prior_min, Prior_max

        if prior.lower() == "median":
            prior = Prior_median(**prior_kwargs)
        elif prior.lower() == "mean":
            prior = Prior_mean(**prior_kwargs)
        elif prior.lower() == "min":
            prior = Prior_min(**prior_kwargs)
        elif prior.lower() == "max":
            prior = Prior_max(**prior_kwargs)
    # Construct the kernel class object
    from ..kernel.se import SE

    kernel = SE(
        use_fingerprint=use_fingerprint,
        use_derivatives=use_derivatives,
        dtype=dtype,
        **kernel_kwargs,
    )
    # Set the hyperparameter optimization method
    if global_optimization:
        # Set global optimization with or without parallelization
        from ..optimizers.globaloptimizer import FactorizedOptimizer

        # Set the line searcher for the hyperparameter optimization
        if parallel:
            from ..optimizers.linesearcher import FineGridSearch

            lineoptimizer_kwargs_default = dict(
                optimize=True,
                multiple_min=False,
                ngrid=80,
                loops=3,
            )
            lineoptimizer_kwargs_default.update(lineoptimizer_kwargs)
            line_optimizer = FineGridSearch(
                parallel=True,
                dtype=dtype,
                **lineoptimizer_kwargs_default,
            )
        else:
            from ..optimizers.linesearcher import GoldenSearch

            lineoptimizer_kwargs_default = dict(
                optimize=True,
                multiple_min=False,
            )
            lineoptimizer_kwargs_default.update(lineoptimizer_kwargs)
            line_optimizer = GoldenSearch(
                parallel=False,
                dtype=dtype,
                **lineoptimizer_kwargs_default,
            )
        # Set the optimizer for the hyperparameter optimization
        optimizer_kwargs_default = dict(
            ngrid=80,
            calculate_init=False,
        )
        optimizer_kwargs_default.update(optimizer_kwargs)
        optimizer = FactorizedOptimizer(
            line_optimizer=line_optimizer,
            parallel=parallel,
            dtype=dtype,
            **optimizer_kwargs_default,
        )
    else:
        from ..optimizers.localoptimizer import ScipyOptimizer

        optimizer_kwargs_default = dict(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        optimizer_kwargs_default.update(optimizer_kwargs)
        # Make the local optimizer
        optimizer = ScipyOptimizer(
            dtype=dtype,
            **optimizer_kwargs_default,
        )
        if parallel:
            warnings.warn(
                "Parallel optimization is not implemented"
                "with local optimization!"
            )
    # Use either the Student t process or the Gaussian process
    model_kwargs.update(kwargs)
    if model.lower() == "tp":
        # Set model
        from ..models.tp import TProcess

        model_kwargs_default = dict(
            a=1e-4,
            b=4.0,
        )
        model_kwargs_default.update(model_kwargs)
        model = TProcess(
            prior=prior,
            kernel=kernel,
            use_derivatives=use_derivatives,
            dtype=dtype,
            **model_kwargs_default,
        )
        # Set objective function
        if global_optimization:
            from ..objectivefunctions.tp.factorized_likelihood import (
                FactorizedLogLikelihood,
            )

            func = FactorizedLogLikelihood(dtype=dtype, **function_kwargs)
        else:
            from ..objectivefunctions.tp.likelihood import LogLikelihood

            func = LogLikelihood(dtype=dtype, **function_kwargs)
    else:
        # Set model
        from ..models.gp import GaussianProcess

        model = GaussianProcess(
            prior=prior,
            kernel=kernel,
            use_derivatives=use_derivatives,
            dtype=dtype,
            **model_kwargs,
        )
        # Set objective function
        if global_optimization:
            from ..objectivefunctions.gp.factorized_likelihood import (
                FactorizedLogLikelihood,
            )

            func = FactorizedLogLikelihood(dtype=dtype, **function_kwargs)
        else:
            from ..objectivefunctions.gp.likelihood import LogLikelihood

            func = LogLikelihood(dtype=dtype, **function_kwargs)
    # Set hpfitter and whether a maximum data set size is applied
    if n_reduced is None:
        from ..hpfitter import HyperparameterFitter

        hpfitter = HyperparameterFitter(
            func=func,
            optimizer=optimizer,
            round_hp=round_hp,
            dtype=dtype,
            **hpfitter_kwargs,
        )
    else:
        from ..hpfitter.redhpfitter import ReducedHyperparameterFitter

        hpfitter = ReducedHyperparameterFitter(
            func=func,
            optimizer=optimizer,
            opt_tr_size=n_reduced,
            round_hp=round_hp,
            dtype=dtype,
            **hpfitter_kwargs,
        )
    model.update_arguments(hpfitter=hpfitter)
    # Set the seed for the model
    if seed is not None:
        model.set_seed(seed=seed)
    # Return the model
    return model


def get_default_database(
    fp=None,
    use_derivatives=True,
    database_reduction=False,
    round_targets=5,
    seed=None,
    dtype=float,
    **database_kwargs,
):
    """
    Get the default Database from the simple given arguments.

    Parameters:
        fp: Fingerprint class object or None
            The fingerprint object used to generate the fingerprints.
            Cartesian coordinates are used if it is None.
        use_derivatives: bool
            Whether to use derivatives of the targets.
        database_reduction: bool
            Whether to used a reduced database after a number
            of training points.
        round_targets: int (optional)
            The number of decimals to round the targets to.
            If None, the targets are not rounded.
        seed: int (optional)
            The random seed for the optimization.
            The seed an also be a RandomState or Generator instance.
            If not given, the default random number generator is used.
        dtype: type
            The data type of the arrays.
        database_kwargs: dict (optional)
            A dictionary with additional arguments for the database.
            Also used for the reduced databases.

    Returns:
        database: Database object
            The Database object with ASE atoms.
    """
    # Set a fingerprint
    if fp is None:
        from ..fingerprint.cartesian import Cartesian

        # Use cartesian coordinates as the fingerprint
        fp = Cartesian(reduce_dimensions=True, use_derivatives=use_derivatives)
        use_fingerprint = False
    else:
        use_fingerprint = True
    # Make the data base ready
    if isinstance(database_reduction, str):
        data_kwargs = dict(
            fingerprint=fp,
            reduce_dimensions=True,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            npoints=50,
            initial_indices=[0, 1],
            include_last=1,
        )
        data_kwargs.update(database_kwargs)
        if database_reduction.lower() == "distance":
            from .database_reduction import DatabaseDistance

            database = DatabaseDistance(**data_kwargs)
        elif database_reduction.lower() == "random":
            from .database_reduction import DatabaseRandom

            database = DatabaseRandom(**data_kwargs)
        elif database_reduction.lower() == "hybrid":
            from .database_reduction import DatabaseHybrid

            database = DatabaseHybrid(**data_kwargs)
        elif database_reduction.lower() == "min":
            from .database_reduction import DatabaseMin

            database = DatabaseMin(**data_kwargs)
        elif database_reduction.lower() == "last":
            from .database_reduction import DatabaseLast

            database = DatabaseLast(**data_kwargs)
        elif database_reduction.lower() == "restart":
            from .database_reduction import DatabaseRestart

            database = DatabaseRestart(**data_kwargs)
        elif database_reduction.lower() == "interest":
            from .database_reduction import DatabasePointsInterest

            database = DatabasePointsInterest(**data_kwargs)
        elif database_reduction.lower() == "each_interest":
            from .database_reduction import DatabasePointsInterestEach

            database = DatabasePointsInterestEach(**data_kwargs)
    else:
        from .database import Database

        data_kwargs = dict(
            reduce_dimensions=True,
        )
        data_kwargs.update(database_kwargs)
        database = Database(
            fingerprint=fp,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=round_targets,
            seed=seed,
            dtype=dtype,
            **data_kwargs,
        )
    return database


def get_default_mlmodel(
    model="tp",
    fp=None,
    baseline=None,
    optimize_hp=True,
    use_pdis=True,
    pdis=None,
    prior="median",
    use_derivatives=True,
    global_optimization=True,
    parallel=False,
    n_reduced=None,
    round_hp=3,
    all_model_kwargs={},
    database_reduction=False,
    round_targets=5,
    database_kwargs={},
    use_ensemble=False,
    verbose=False,
    seed=None,
    dtype=float,
    **kwargs,
):
    """
    Get the default ML model with a database for the ASE Atoms
    from the simple given arguments.

    Parameters:
        model: str
            Either the tp that gives the Studen T process or
            gp that gives the Gaussian process.
        fp: Fingerprint class object or None
            The fingerprint object used to generate the fingerprints.
            Cartesian coordinates are used if it is None.
        baseline: Baseline object
            The Baseline object calculator that calculates energy and forces.
        optimize_hp: bool
            Whether to optimize the hyperparameters when the model is trained.
        use_pdis: bool
            Whether to make prior distributions for the hyperparameters.
        pdis: dict (optional)
            A dict of prior distributions for each hyperparameter type.
            If None, the default prior distributions are used.
            No prior distributions are used if use_pdis=False or pdis is {}.
        prior: str
            Specify what prior mean should be used.
        use_derivatives: bool
            Whether to use derivatives of the targets.
        global_optimization: bool
            Whether to perform a global optimization of the hyperparameters.
            A local optimization is used if global_optimization=False,
            which can not be parallelized.
        parallel: bool
            Whether to optimize the hyperparameters in parallel.
        n_reduced: int or None
            If n_reduced is an integer, the hyperparameters are only optimized
                when the data set size is equal to or below the integer.
            If n_reduced is None, the hyperparameter is always optimized.
        round_hp: int (optional)
            The number of decimals to round the hyperparameters to.
            If None, the hyperparameters are not rounded.
        all_model_kwargs: dict (optional)
            A dictionary with additional arguments for the model.
            It also can include model_kwargs, prior_kwargs,
            kernel_kwargs, hpfitter_kwargs, optimizer_kwargs,
            lineoptimizer_kwargs, and function_kwargs.
        database_reduction: bool
            Whether to used a reduced database after a number
            of training points.
        round_targets: int (optional)
            The number of decimals to round the targets to.
            If None, the targets are not rounded.
        database_kwargs: dict
            A dictionary with the arguments for the database
            if it is used.
        verbose: bool
            Whether to print statements in the optimization.
        seed: int (optional)
            The random seed for the optimization.
            The seed an also be a RandomState or Generator instance.
            If not given, the default random number generator is used.
        dtype: type
            The data type of the arrays.
        kwargs: dict (optional)
            Additional keyword arguments for the MLModel class.

    Returns:
        mlmodel: MLModel class object
            Machine Learning model used for ASE Atoms and calculator.
    """
    from .mlmodel import MLModel

    # Check if fingerprints are used
    if fp is None:
        use_fingerprint = False
    else:
        use_fingerprint = True
    # Make the model
    if isinstance(model, str):
        model = get_default_model(
            model=model,
            prior=prior,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            global_optimization=global_optimization,
            parallel=parallel,
            n_reduced=n_reduced,
            round_hp=round_hp,
            seed=seed,
            dtype=dtype,
            **all_model_kwargs,
        )
    # Make the database
    database = get_default_database(
        fp=fp,
        use_derivatives=use_derivatives,
        database_reduction=database_reduction,
        round_targets=round_targets,
        seed=seed,
        dtype=dtype,
        **database_kwargs,
    )
    # Make prior distributions for the hyperparameters if specified
    if use_pdis and pdis is None:
        from ..pdistributions.normal import Normal_prior

        pdis = dict(
            length=Normal_prior(mu=[-0.8], std=[0.2], dtype=dtype),
            noise=Normal_prior(mu=[-9.0], std=[1.0], dtype=dtype),
        )
    elif not use_pdis:
        pdis = None
    # Make the ML model with database
    return MLModel(
        model=model,
        database=database,
        baseline=baseline,
        optimize=optimize_hp,
        pdis=pdis,
        verbose=verbose,
        dtype=dtype,
        **kwargs,
    )
