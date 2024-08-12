import unittest
import numpy as np
from .functions import create_func, make_train_test_set, check_minima


class TestGPOptimizer(unittest.TestCase):
    """
    Test if the Gaussian Process can be optimized with
    all existing optimization methods and objective functions.
    """

    def test_function(self):
        "Test if the function value of the LogLikelihood is obtainable."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import FunctionEvaluation
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=FunctionEvaluation(jac=True),
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol = gp.optimize(
            x_tr,
            f_tr,
            retrain=False,
            hp=None,
            pdis=None,
            verbose=False,
        )
        # Test the solution is correct
        self.assertTrue(abs(sol["fun"] - 393.422) < 1e-2)

    def test_local_jac(self):
        """
        Test if the GP can be local optimized with gradients
        wrt. the hyperparameters.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import (
            LogLikelihood,
        )
        from catlearn.regression.gp.hpfitter import HyperparameterFitter

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the optimizer
        optimizer = ScipyOptimizer(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol = gp.optimize(
            x_tr,
            f_tr,
            retrain=False,
            hp=None,
            pdis=None,
            verbose=False,
        )
        # Test the solution is a minimum
        is_minima = check_minima(
            sol,
            x_tr,
            f_tr,
            gp,
            pdis=None,
            is_model_gp=True,
        )
        self.assertTrue(is_minima)

    def test_local_nojac(self):
        """
        Test if the GP can be local optimized without gradients
        wrt. the hyperparameters.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the optimizer
        optimizer = ScipyOptimizer(
            maxiter=500,
            jac=False,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol = gp.optimize(
            x_tr,
            f_tr,
            retrain=False,
            hp=None,
            pdis=None,
            verbose=False,
        )
        # Test the solution is a minimum
        is_minima = check_minima(
            sol,
            x_tr,
            f_tr,
            gp,
            pdis=None,
            is_model_gp=True,
        )
        self.assertTrue(is_minima)

    def test_local_prior(self):
        "Test if the GP can be local optimized with prior distributions."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyPriorOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.pdistributions import Normal_prior

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the optimizer
        optimizer = ScipyPriorOptimizer(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Construct the prior distributions of the hyperparameters
        pdis = dict(
            length=Normal_prior(mu=0.0, std=2.0),
            noise=Normal_prior(mu=-4.0, std=2.0),
        )
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol = gp.optimize(
            x_tr,
            f_tr,
            retrain=False,
            hp=None,
            pdis=pdis,
            verbose=False,
        )
        # Test the solution is a minimum
        # The prior is None since the ScipyPriorOptimizer is used
        is_minima = check_minima(
            sol,
            x_tr,
            f_tr,
            gp,
            pdis=pdis,
            is_model_gp=True,
        )
        self.assertTrue(is_minima)

    def test_local_ed_guess(self):
        "Test if the GP can be local optimized with educated guesses."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyGuessOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary.strict import StrictBoundaries

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the optimizer
        optimizer = ScipyGuessOptimizer(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
            bounds=StrictBoundaries(),
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol = gp.optimize(
            x_tr,
            f_tr,
            retrain=False,
            hp=None,
            pdis=None,
            verbose=False,
        )
        # Test the solution is a minimum
        is_minima = check_minima(
            sol,
            x_tr,
            f_tr,
            gp,
            pdis=None,
            is_model_gp=True,
        )
        self.assertTrue(is_minima)

    def test_random(self):
        "Test if the GP can be local optimized from random sampling."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            ScipyOptimizer,
            RandomSamplingOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
            EducatedBoundaries,
            VariableTransformation,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the local optimizer
        local_optimizer = ScipyOptimizer(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Make the global optimizer
        optimizer = RandomSamplingOptimizer(
            local_optimizer=local_optimizer,
            maxiter=600,
            npoints=10,
            parallel=False,
        )
        # Make fixed boundary conditions for one of the tests
        bounds_dict = dict(
            length=[[-3.0, 3.0]],
            noise=[[-8.0, 0.0]],
            prefactor=[[-2.0, 4.0]],
        )
        # Define test list of arguments for the random sampling optimizer
        bounds_list = [
            VariableTransformation(bounds=None),
            EducatedBoundaries(),
            HPBoundaries(bounds_dict=bounds_dict),
        ]
        # Test the arguments for the random sampling optimizer
        for bounds in bounds_list:
            with self.subTest(bounds=bounds):
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=LogLikelihood(),
                    optimizer=optimizer,
                    bounds=bounds,
                )
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol = gp.optimize(
                    x_tr,
                    f_tr,
                    retrain=False,
                    hp=None,
                    pdis=None,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=None,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)

    def test_grid(self):
        "Test if the GP can be optimized from grid search."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            ScipyOptimizer,
            GridOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
            EducatedBoundaries,
            VariableTransformation,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the local optimizer
        local_optimizer = ScipyOptimizer(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Make the global optimizer kwargs
        opt_kwargs = dict(maxiter=500, n_each_dim=5, parallel=False)
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation(bounds=None)
        bounds_ed = EducatedBoundaries()
        fixed_bounds = HPBoundaries(
            bounds_dict=dict(
                length=[[-3.0, 3.0]],
                noise=[[-8.0, 0.0]],
                prefactor=[[-2.0, 4.0]],
            ),
            log=True,
        )
        # Define test list of arguments for the grid search optimizer
        test_kwargs = [
            (
                bounds_trans,
                GridOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=False,
                    **opt_kwargs
                ),
            ),
            (
                bounds_trans,
                GridOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=True,
                    **opt_kwargs
                ),
            ),
            (
                bounds_ed,
                GridOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=True,
                    **opt_kwargs
                ),
            ),
            (
                fixed_bounds,
                GridOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=True,
                    **opt_kwargs
                ),
            ),
        ]
        # Test the arguments for the grid search optimizer
        for bounds, optimizer in test_kwargs:
            with self.subTest(bounds=bounds, optimizer=optimizer):
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=LogLikelihood(),
                    optimizer=optimizer,
                    bounds=bounds,
                )
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol = gp.optimize(
                    x_tr,
                    f_tr,
                    retrain=False,
                    hp=None,
                    pdis=None,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=None,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)

    def test_line(self):
        "Test if the GP can be optimized from iteratively line search."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            ScipyOptimizer,
            IterativeLineOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
            EducatedBoundaries,
            VariableTransformation,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the local optimizer
        local_optimizer = ScipyOptimizer(
            maxiter=500,
            jac=True,
            method="l-bfgs-b",
            use_bounds=False,
            tol=1e-12,
        )
        # Make the global optimizer kwargs
        opt_kwargs = dict(maxiter=500, n_each_dim=10, loops=3, parallel=False)
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation(bounds=None)
        bounds_ed = EducatedBoundaries()
        fixed_bounds = HPBoundaries(
            bounds_dict=dict(
                length=[[-3.0, 3.0]],
                noise=[[-8.0, 0.0]],
                prefactor=[[-2.0, 4.0]],
            ),
            log=True,
        )
        # Define test list of arguments for the line search optimizer
        test_kwargs = [
            (
                bounds_trans,
                IterativeLineOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=False,
                    **opt_kwargs
                ),
            ),
            (
                bounds_trans,
                IterativeLineOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=True,
                    **opt_kwargs
                ),
            ),
            (
                bounds_ed,
                IterativeLineOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=True,
                    **opt_kwargs
                ),
            ),
            (
                fixed_bounds,
                IterativeLineOptimizer(
                    local_optimizer=local_optimizer,
                    optimize=True,
                    **opt_kwargs
                ),
            ),
        ]
        # Test the arguments for the line search optimizer
        for bounds, optimizer in test_kwargs:
            with self.subTest(bounds=bounds, optimizer=optimizer):
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=LogLikelihood(),
                    optimizer=optimizer,
                    bounds=bounds,
                )
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol = gp.optimize(
                    x_tr,
                    f_tr,
                    retrain=False,
                    hp=None,
                    pdis=None,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=None,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)

    def test_basin(self):
        "Test if the GP can be optimized from basin hopping."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import BasinOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the dictionary of the optimization
        local_kwargs = dict(tol=1e-12, method="L-BFGS-B")
        opt_kwargs = dict(
            niter=5,
            interval=10,
            T=1.0,
            stepsize=0.1,
            niter_success=None,
        )
        # Make the optimizer
        optimizer = BasinOptimizer(
            maxiter=500,
            jac=True,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol = gp.optimize(
            x_tr,
            f_tr,
            retrain=False,
            hp=None,
            pdis=None,
            verbose=False,
        )
        # Test the solution deviation
        self.assertTrue(abs(sol["fun"] - 47.042) < 1e-2)
        self.assertTrue(
            np.linalg.norm(sol["x"] - np.array([1.97, -15.43, 1.79])) < 1e-2
        )
        # Test the solution is a minimum
        is_minima = check_minima(
            sol,
            x_tr,
            f_tr,
            gp,
            pdis=None,
            is_model_gp=True,
        )
        self.assertTrue(is_minima)

    def test_annealling(self):
        "Test if the GP can be optimized from simulated annealling."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import AnneallingOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
            EducatedBoundaries,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the dictionary of the optimization
        local_kwargs = dict(tol=1e-12, method="L-BFGS-B")
        opt_kwargs = dict(
            initial_temp=5230.0,
            restart_temp_ratio=2e-05,
            visit=2.62,
            accept=-5.0,
            seed=None,
            no_local_search=False,
        )
        # Make the optimizer
        optimizer = AnneallingOptimizer(
            maxiter=500,
            jac=False,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
        )
        # Make the boundary conditions for the tests
        bounds_ed = EducatedBoundaries()
        fixed_bounds = HPBoundaries(
            bounds_dict=dict(
                length=[[-3.0, 3.0]],
                noise=[[-8.0, 0.0]],
                prefactor=[[-2.0, 4.0]],
            ),
            log=True,
        )
        # Define test list of arguments for the simulated annealling optimizer
        bounds_list = [bounds_ed, fixed_bounds]
        # Test the arguments for the simulated annealling optimizer
        for bounds in bounds_list:
            with self.subTest(bounds=bounds):
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=LogLikelihood(),
                    optimizer=optimizer,
                    bounds=bounds,
                )
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol = gp.optimize(
                    x_tr,
                    f_tr,
                    retrain=False,
                    hp=None,
                    pdis=None,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=None,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)

    def test_annealling_trans(self):
        """
        Test if the GP can be optimized from simulated annealling
        with variable transformation.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import AnneallingTransOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
            VariableTransformation,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the dictionary of the optimization
        local_kwargs = dict(tol=1e-12, method="L-BFGS-B")
        opt_kwargs = dict(
            initial_temp=5230.0,
            restart_temp_ratio=2e-05,
            visit=2.62,
            accept=-5.0,
            seed=None,
            no_local_search=False,
        )
        # Make the optimizer
        optimizer = AnneallingTransOptimizer(
            maxiter=500,
            jac=False,
            opt_kwargs=opt_kwargs,
            local_kwargs=local_kwargs,
        )
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation(bounds=None)
        fixed_bounds = HPBoundaries(
            bounds_dict=dict(
                length=[[-3.0, 3.0]],
                noise=[[-8.0, 0.0]],
                prefactor=[[-2.0, 4.0]],
            ),
            log=True,
        )
        bounds_fixed_trans = VariableTransformation(bounds=fixed_bounds)
        # Define test list of arguments for the simulated annealling optimizer
        bounds_list = [bounds_trans, bounds_fixed_trans]
        # Test the arguments for the simulated annealling optimizer
        for bounds in bounds_list:
            with self.subTest(bounds=bounds):
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=LogLikelihood(),
                    optimizer=optimizer,
                    bounds=bounds,
                )
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol = gp.optimize(
                    x_tr,
                    f_tr,
                    retrain=False,
                    hp=None,
                    pdis=None,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=None,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)

    def test_line_search_scale(self):
        """
        Test if the GP can be optimized from line search
        in the length-scale hyperparameter.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            FactorizedOptimizer,
            GoldenSearch,
            FineGridSearch,
            TransGridSearch,
        )
        from catlearn.regression.gp.objectivefunctions.gp import (
            FactorizedLogLikelihood,
        )
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
            EducatedBoundaries,
            VariableTransformation,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the dictionary of the optimization
        opt_kwargs = dict(maxiter=500, jac=False, tol=1e-5, parallel=False)
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation(bounds=None)
        bounds_ed = EducatedBoundaries()
        fixed_bounds = HPBoundaries(
            bounds_dict=dict(
                length=[[-3.0, 3.0]],
                noise=[[-8.0, 0.0]],
                prefactor=[[-2.0, 4.0]],
            ),
            log=True,
        )
        # Define test list of arguments for the line search optimizer
        test_kwargs = [
            (
                bounds_ed,
                GoldenSearch(optimize=False, multiple_min=False, **opt_kwargs),
            ),
            (
                bounds_trans,
                GoldenSearch(optimize=False, multiple_min=False, **opt_kwargs),
            ),
            (
                bounds_trans,
                GoldenSearch(optimize=True, multiple_min=False, **opt_kwargs),
            ),
            (
                bounds_trans,
                GoldenSearch(optimize=True, multiple_min=True, **opt_kwargs),
            ),
            (
                bounds_trans,
                FineGridSearch(
                    optimize=True,
                    multiple_min=False,
                    loops=3,
                    ngrid=80,
                    **opt_kwargs
                ),
            ),
            (
                bounds_trans,
                FineGridSearch(
                    optimize=True,
                    multiple_min=True,
                    loops=3,
                    ngrid=80,
                    **opt_kwargs
                ),
            ),
            (
                fixed_bounds,
                FineGridSearch(
                    optimize=True,
                    multiple_min=True,
                    loops=3,
                    ngrid=80,
                    **opt_kwargs
                ),
            ),
            (
                bounds_trans,
                TransGridSearch(
                    optimize=True,
                    use_likelihood=False,
                    loops=3,
                    ngrid=80,
                    **opt_kwargs
                ),
            ),
            (
                bounds_trans,
                TransGridSearch(
                    optimize=True,
                    use_likelihood=True,
                    loops=3,
                    ngrid=80,
                    **opt_kwargs
                ),
            ),
        ]
        # Test the arguments for the line search optimizer
        for bounds, line_optimizer in test_kwargs:
            with self.subTest(bounds=bounds, line_optimizer=line_optimizer):
                # Make the optimizer
                optimizer = FactorizedOptimizer(
                    line_optimizer=line_optimizer,
                    bounds=bounds,
                    ngrid=80,
                )
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=FactorizedLogLikelihood(),
                    optimizer=optimizer,
                )
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol = gp.optimize(
                    x_tr,
                    f_tr,
                    retrain=False,
                    hp=None,
                    pdis=None,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=None,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)


if __name__ == "__main__":
    unittest.main()
