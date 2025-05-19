import unittest
from .functions import create_func, make_train_test_set, check_minima


class TestGPOptimizerASEParallel(unittest.TestCase):
    """
    Test if the Gaussian Process can be optimized with
    all existing optimization methods and objective functions
    that works in parallel with ASE.
    """

    def test_random(self):
        """
        Test if the GP can be local optimized from random sampling in parallel.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            ScipyOptimizer,
            RandomSamplingOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, _, _ = make_train_test_set(
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
        )
        # Make the global optimizer
        optimizer = RandomSamplingOptimizer(
            local_optimizer=local_optimizer,
            maxiter=600,
            npoints=12,
            parallel=True,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        gp.set_seed(seed=seed)
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
        "Test if the GP can be brute-force grid optimized in parallel."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            ScipyOptimizer,
            GridOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary.hptrans import (
            VariableTransformation,
        )

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, _, _ = make_train_test_set(
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
        )
        # Make the global optimizer
        optimizer = GridOptimizer(
            local_optimizer=local_optimizer,
            optimize=False,
            maxiter=500,
            n_each_dim=5,
            parallel=True,
        )
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation()
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
            bounds=bounds_trans,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        gp.set_seed(seed=seed)
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
        "Test if the GP can be iteratively line search optimized in parallel."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            ScipyOptimizer,
            IterativeLineOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import VariableTransformation

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
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
        )
        # Make the global optimizer
        optimizer = IterativeLineOptimizer(
            local_optimizer=local_optimizer,
            optimize=False,
            maxiter=500,
            n_each_dim=10,
            loops=3,
            parallel=True,
        )
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation()
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
            bounds=bounds_trans,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        gp.set_seed(seed=seed)
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
        Test if the GP can be optimized from line search in
        the length-scale hyperparameter in parallel.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            FineGridSearch,
            FactorizedOptimizer,
        )
        from catlearn.regression.gp.objectivefunctions.gp import (
            FactorizedLogLikelihood,
        )
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.hpboundary import VariableTransformation

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = False
        x_tr, f_tr, _, _ = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the boundary conditions for the tests
        bounds_trans = VariableTransformation()
        # Make the line optimizer
        line_optimizer = FineGridSearch(
            optimize=True,
            multiple_min=False,
            loops=3,
            ngrid=80,
            maxiter=500,
            jac=False,
            tol=1e-5,
            parallel=True,
        )
        # Make the global optimizer
        optimizer = FactorizedOptimizer(
            line_optimizer=line_optimizer,
            ngrid=80,
            parallel=True,
        )
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=FactorizedLogLikelihood(),
            optimizer=optimizer,
            bounds=bounds_trans,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        gp.set_seed(seed=seed)
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
