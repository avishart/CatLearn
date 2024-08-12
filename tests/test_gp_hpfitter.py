import unittest
import numpy as np
from .functions import create_func, make_train_test_set, check_minima


class TestGPHpfitter(unittest.TestCase):
    """
    Test if the hyperparameters of the Gaussian Process can be optimized
    with hyperparameter fitters.
    """

    def test_hpfitters_noderiv(self):
        """
        Test if the hyperparameters of the GP without derivatives
        can be optimized.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import (
            HyperparameterFitter,
            ReducedHyperparameterFitter,
            FBPMGP,
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
            te=20,
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
        # Define the list of hyperparameter fitter objects that are tested
        hpfitter_list = [
            HyperparameterFitter(func=LogLikelihood(), optimizer=optimizer),
            HyperparameterFitter(
                func=LogLikelihood(), optimizer=optimizer, use_stored_sols=True
            ),
            ReducedHyperparameterFitter(
                func=LogLikelihood(), opt_tr_size=50, optimizer=optimizer
            ),
            ReducedHyperparameterFitter(
                func=LogLikelihood(), opt_tr_size=10, optimizer=optimizer
            ),
            FBPMGP(Q=None, n_test=50, ngrid=80, bounds=None),
        ]
        # Test the hyperparameter fitter objects
        for index, hpfitter in enumerate(hpfitter_list):
            with self.subTest(hpfitter=hpfitter):
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
                if index < 3:
                    is_minima = check_minima(
                        sol, x_tr, f_tr, gp, pdis=None, is_model_gp=True
                    )
                    self.assertTrue(is_minima)
                elif index == 4:
                    self.assertTrue(abs(sol["fun"] - 0.883) < 1e-2)

    def test_hpfitters_deriv(self):
        """
        Test if the hyperparameters of the GP with derivatives
        can be optimized.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import (
            HyperparameterFitter,
            ReducedHyperparameterFitter,
            FBPMGP,
        )

        # Create the data set
        x, f, g = create_func()
        # Whether to learn from the derivatives
        use_derivatives = True
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=20,
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
        # Define the list of hyperparameter fitter objects that are tested
        hpfitter_list = [
            HyperparameterFitter(func=LogLikelihood(), optimizer=optimizer),
            HyperparameterFitter(
                func=LogLikelihood(),
                optimizer=optimizer,
                use_stored_sols=True,
            ),
            ReducedHyperparameterFitter(
                func=LogLikelihood(),
                opt_tr_size=50,
                optimizer=optimizer,
            ),
            ReducedHyperparameterFitter(
                func=LogLikelihood(),
                opt_tr_size=10,
                optimizer=optimizer,
            ),
            FBPMGP(Q=None, n_test=50, ngrid=80, bounds=None),
        ]
        # Test the hyperparameter fitter objects
        for index, hpfitter in enumerate(hpfitter_list):
            with self.subTest(hpfitter=hpfitter):
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
                if index < 3:
                    is_minima = check_minima(
                        sol,
                        x_tr,
                        f_tr,
                        gp,
                        pdis=None,
                        is_model_gp=True,
                    )
                    self.assertTrue(is_minima)
                elif index == 4:
                    self.assertTrue(abs(sol["fun"] - (-8.171)) < 1e-2)


if __name__ == "__main__":
    unittest.main()
