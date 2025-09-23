import unittest
from .functions import create_func, make_train_test_set, check_minima


class TestTPObjectiveFunctions(unittest.TestCase):
    """
    Test if the Student T Process can be optimized with
    all existing objective functions.
    """

    def test_local(self):
        """
        Test if the TP can be local optimized with all objective functions
        that does not use eigendecomposition.
        """
        from catlearn.regression.gp.models import TProcess
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.objectivefunctions.tp import LogLikelihood

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
        # Construct the hyperparameter fitter
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
        )
        # Construct the Student t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            hpfitter=hpfitter,
            use_derivatives=use_derivatives,
        )
        # Set random seed to give the same results every time
        tp.set_seed(seed=seed)
        # Optimize the hyperparameters
        sol = tp.optimize(
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
            tp,
            pdis=None,
            func=None,
            is_model_gp=False,
        )
        self.assertTrue(is_minima)

    def test_line_search_scale(self):
        """
        Test if the TP can be optimized from line search in
        the length-scale hyperparameter with all objective functions
        that uses eigendecomposition.
        """
        from catlearn.regression.gp.models import TProcess
        from catlearn.regression.gp.optimizers import (
            FactorizedOptimizer,
            FineGridSearch,
            NoiseGrid,
            NoiseGoldenSearch,
            NoiseFineGridSearch,
        )
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.objectivefunctions.tp import (
            FactorizedLogLikelihood,
            FactorizedLogLikelihoodSVD,
        )
        from catlearn.regression.gp.hpboundary import (
            HPBoundaries,
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
        # Make the default boundaries for the hyperparameters
        default_bounds = VariableTransformation()
        # Make fixed boundary conditions for one of the tests
        fixed_bounds = HPBoundaries(
            bounds_dict=dict(
                length=[[-1.0, 3.0]],
                noise=[[-4.0, -1.0]],
            ),
            log=True,
        )
        # Make the optimizers
        line_optimizer = FineGridSearch(
            tol=1e-5,
            loops=3,
            ngrid=80,
            optimize=True,
            multiple_min=False,
        )
        optimizer = FactorizedOptimizer(
            line_optimizer=line_optimizer,
            maxiter=500,
            ngrid=80,
            parallel=False,
        )
        # Define the list of objective function objects that are tested
        obj_list = [
            (
                default_bounds,
                FactorizedLogLikelihood(
                    modification=False,
                    ngrid=250,
                    noise_optimizer=NoiseGrid(),
                ),
            ),
            (
                default_bounds,
                FactorizedLogLikelihood(
                    modification=True,
                    ngrid=250,
                    noise_optimizer=NoiseGrid(),
                ),
            ),
            (
                default_bounds,
                FactorizedLogLikelihood(
                    modification=False,
                    ngrid=80,
                    noise_optimizer=NoiseGoldenSearch(),
                ),
            ),
            (
                default_bounds,
                FactorizedLogLikelihood(
                    modification=False,
                    ngrid=80,
                    noise_optimizer=NoiseFineGridSearch(),
                ),
            ),
            (
                fixed_bounds,
                FactorizedLogLikelihood(
                    modification=False,
                    ngrid=250,
                    noise_optimizer=NoiseGrid(),
                ),
            ),
            (
                default_bounds,
                FactorizedLogLikelihoodSVD(
                    modification=False,
                    ngrid=250,
                    noise_optimizer=NoiseGrid(),
                ),
            ),
        ]
        # Test the objective function objects
        for bounds, obj_func in obj_list:
            with self.subTest(bounds=bounds, obj_func=obj_func):
                # Construct the hyperparameter fitter
                hpfitter = HyperparameterFitter(
                    func=obj_func,
                    optimizer=optimizer,
                    bounds=bounds,
                )
                # Construct the Student t process
                tp = TProcess(
                    hp=dict(length=[2.0], noise=[-5.0]),
                    hpfitter=hpfitter,
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                tp.set_seed(seed=seed)
                # Optimize the hyperparameters
                sol = tp.optimize(
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
                    tp,
                    pdis=None,
                    func=obj_func,
                    is_model_gp=False,
                )
                self.assertTrue(is_minima)


if __name__ == "__main__":
    unittest.main()
