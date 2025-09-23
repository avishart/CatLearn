import unittest
from .functions import create_func, make_train_test_set, check_minima


class TestGPPdis(unittest.TestCase):
    """
    Test if the Gaussian Process can be optimized with
    all existing prior distributions of the hyperparameters.
    """

    def test_local_prior(self):
        "Test if the GP can be local optimized with prior distributions."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.pdistributions import (
            Uniform_prior,
            Normal_prior,
            Gen_normal_prior,
            Gamma_prior,
            Invgamma_prior,
        )
        from catlearn.regression.gp.hpboundary import StrictBoundaries

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
        # Make the optimizer
        optimizer = ScipyOptimizer(
            maxiter=500,
            jac=True,
        )
        # Define the list of prior distribution objects that are tested
        test_pdis = [
            (False, Uniform_prior(start=[-9.0], end=[9.0], prob=[1.0 / 18.0])),
            (True, Uniform_prior(start=[-9.0], end=[9.0], prob=[1.0 / 18.0])),
            (False, Normal_prior(mu=[0.0], std=[3.0])),
            (True, Normal_prior(mu=[0.0], std=[3.0])),
            (False, Gen_normal_prior(mu=[0.0], s=[3.0], v=[2])),
            (True, Gen_normal_prior(mu=[0.0], s=[3.0], v=[2])),
            (False, Gamma_prior(a=[1e-5], b=[1e-5])),
            (True, Gamma_prior(a=[1e-5], b=[1e-5])),
            (False, Invgamma_prior(a=[1e-5], b=[1e-5])),
            (True, Invgamma_prior(a=[1e-5], b=[1e-5])),
        ]
        # Test the prior distributions
        for use_update_pdis, pdis_d in test_pdis:
            with self.subTest(use_update_pdis=use_update_pdis, pdis_d=pdis_d):
                # Construct the prior distribution objects
                pdis = dict(length=pdis_d.copy(), noise=pdis_d.copy())
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=LogLikelihood(),
                    optimizer=optimizer,
                    bounds=StrictBoundaries(),
                    use_update_pdis=use_update_pdis,
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
                    pdis=pdis,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=pdis,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)

    def test_global_prior(self):
        "Test if the GP can be global optimized with prior distributions."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.optimizers import (
            FactorizedOptimizer,
            FineGridSearch,
        )
        from catlearn.regression.gp.objectivefunctions.gp import (
            FactorizedLogLikelihood,
        )
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.pdistributions import (
            Uniform_prior,
            Normal_prior,
            Gen_normal_prior,
            Gamma_prior,
            Invgamma_prior,
        )
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
        # Make the line optimizer
        line_optimizer = FineGridSearch(
            optimize=True,
            multiple_min=False,
            loops=3,
            ngrid=80,
            maxiter=500,
            jac=False,
            tol=1e-5,
            parallel=False,
        )
        # Make the global optimizer
        optimizer = FactorizedOptimizer(
            line_optimizer=line_optimizer,
            ngrid=80,
            parallel=False,
        )
        # Define the list of prior distribution objects that are tested
        test_pdis = [
            (False, Uniform_prior(start=[-9.0], end=[9.0], prob=[1.0 / 18.0])),
            (True, Uniform_prior(start=[-9.0], end=[9.0], prob=[1.0 / 18.0])),
            (False, Normal_prior(mu=[0.0], std=[3.0])),
            (True, Normal_prior(mu=[0.0], std=[3.0])),
            (False, Gen_normal_prior(mu=[0.0], s=[3.0], v=[2])),
            (True, Gen_normal_prior(mu=[0.0], s=[3.0], v=[2])),
            (False, Gamma_prior(a=[1e-5], b=[1e-5])),
            (True, Gamma_prior(a=[1e-5], b=[1e-5])),
            (False, Invgamma_prior(a=[1e-5], b=[1e-5])),
            (True, Invgamma_prior(a=[1e-5], b=[1e-5])),
        ]
        # Test the prior distributions
        for use_update_pdis, pdis_d in test_pdis:
            with self.subTest(use_update_pdis=use_update_pdis, pdis_d=pdis_d):
                # Construct the prior distribution objects
                pdis = dict(length=pdis_d.copy(), noise=pdis_d.copy())
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=FactorizedLogLikelihood(),
                    optimizer=optimizer,
                    bounds=VariableTransformation(),
                    use_update_pdis=use_update_pdis,
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
                    pdis=pdis,
                    verbose=False,
                )
                # Test the solution is a minimum
                is_minima = check_minima(
                    sol,
                    x_tr,
                    f_tr,
                    gp,
                    pdis=pdis,
                    is_model_gp=True,
                )
                self.assertTrue(is_minima)


if __name__ == "__main__":
    unittest.main()
