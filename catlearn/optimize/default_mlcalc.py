def get_default_mlcalc(model='tp',use_derivatives=True,fp=None,baseline=None,optimize=True,parallelize=False,database_reduction=False,ensemble=False,npoints=50,**kwargs):
    " Get a default ML calculator if a calculator is not given. This is a recommended ML calculator."
    from ..regression.gaussianprocess.calculator.mlcalc import MLCalculator
    from ..regression.gaussianprocess.calculator.mlmodel import MLModel
    from ..regression.gaussianprocess.kernel.se import SE
    from ..regression.gaussianprocess.means import Prior_max
    from ..regression.gaussianprocess.hpfitter.hpfitter import HyperparameterFitter
    from ..regression.gaussianprocess.pdistributions import Normal_prior
    # Set a fingerprint
    if fp is None:
        from ..regression.gaussianprocess.fingerprint.cartesian import Cartesian
        # Use cartesian coordinates as the fingerprint
        fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True)
        use_fingerprint=False
    else:
        use_fingerprint=True
    # Set the hyperparameter optimization method with or without parallelization
    if parallelize:
        from ..regression.gaussianprocess.optimizers.local_opt import fine_grid_search
        from ..regression.gaussianprocess.optimizers.mpi_global_opt import line_search_scale_parallel,calculate_list_values_parallelize
        local_kwargs=dict(fun_list=calculate_list_values_parallelize,tol=1e-5,loops=3,iterloop=80,optimize=True,multiple_min=False)
        kwargs_optimize=dict(local_run=fine_grid_search,maxiter=500,jac=False,ngrid=80,bounds=None,hptrans=True,use_bounds=True,local_kwargs=local_kwargs)
        optimization_method=line_search_scale_parallel
    else:
        from ..regression.gaussianprocess.optimizers.local_opt import run_golden
        from ..regression.gaussianprocess.optimizers.global_opt import line_search_scale
        local_kwargs=dict(tol=1e-5,optimize=True,multiple_min=False)
        kwargs_optimize=dict(local_run=run_golden,maxiter=500,jac=False,ngrid=80,bounds=None,use_bounds=True,local_kwargs=local_kwargs)
        optimization_method=line_search_scale
    # Construct the kernel class object
    kernel=SE(use_fingerprint=use_fingerprint,use_derivatives=use_derivatives)
    # Use either the Student t process or the Gaussian process
    if model.lower()=='tp':
        from ..regression.gaussianprocess.models.tp import TProcess
        from ..regression.gaussianprocess.objectivefunctions.tp.factorized_likelihood import FactorizedLogLikelihood
        hpfitter=HyperparameterFitter(FactorizedLogLikelihood(),optimization_method=optimization_method,opt_kwargs=kwargs_optimize)
        model=TProcess(prior=Prior_max(),kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
    else:
        from ..regression.gaussianprocess.models.gp import GaussianProcess
        from ..regression.gaussianprocess.objectivefunctions.gp.factorized_likelihood import FactorizedLogLikelihood
        hpfitter=HyperparameterFitter(FactorizedLogLikelihood(),optimization_method=optimization_method,opt_kwargs=kwargs_optimize)
        model=GaussianProcess(prior=Prior_max(),kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
    # Use an ensemble model
    if ensemble:
        from ..regression.gaussianprocess.ensemble.ensemble_clustering import EnsembleClustering
        from ..regression.gaussianprocess.ensemble.clustering.k_means_number import K_means_number
        clustering=K_means_number(data_number=npoints,maxiter=20,tol=1e-3)
        model=EnsembleClustering(model=model,clustering=clustering,variance_ensemble=True)
    # Make the data base ready
    if database_reduction:
        from ..regression.gaussianprocess.calculator.database_reduction import DatabaseLast
        database=DatabaseLast(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint,npoints=npoints,initial_indicies=[0,1])
    else:
        from ..regression.gaussianprocess.calculator.database import Database
        database=Database(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint)
    # Make prior distributions for hyperparameters
    pdis=dict(length=Normal_prior(mu=[0.0],std=[2.0]),noise=Normal_prior(mu=[-9.0],std=[1.0]))
    # Make the ML model with model and database
    ml_opt_kwargs=dict(retrain=True,pdis=pdis)
    mlmodel=MLModel(model=model,database=database,baseline=baseline,optimize=optimize,optimize_kwargs=ml_opt_kwargs)
    # Finally make the calculator
    mlcalc=MLCalculator(mlmodel=mlmodel,calculate_uncertainty=True,calculate_forces=True)
    return mlcalc

