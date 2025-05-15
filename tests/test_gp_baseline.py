import unittest
from .functions import create_h2_atoms, make_train_test_set


class TestGPBaseline(unittest.TestCase):
    """
    Test if the baseline can be used in the Gaussian process as
    an ASE calculator with different database forms for ASE atoms.
    """

    def test_predict(self):
        "Test if the GP calculator can predict energy and forces."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.kernel import SE
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.fingerprint import Cartesian
        from catlearn.regression.gp.calculator import (
            Database,
            MLModel,
            MLCalculator,
        )
        from catlearn.regression.gp.baseline import (
            BaselineCalculator,
            BornRepulsionCalculator,
            RepulsionCalculator,
            MieCalculator,
        )

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_h2_atoms(gridsize=50, seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = True
        x_tr, _, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=10,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Make the hyperparameter fitter
        optimizer = ScipyOptimizer(
            maxiter=500,
        )
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
            round_hp=3,
        )
        # Define the list of baseline objects that are tested
        baseline_list = [
            BaselineCalculator(),
            BornRepulsionCalculator(),
            RepulsionCalculator(),
            MieCalculator(),
        ]
        # Make a list of the error values that the test compares to
        error_list = [2.11773, 3.21230, 0.26532, 1.08910]
        # Test the baseline objects
        for index, baseline in enumerate(baseline_list):
            with self.subTest(baseline=baseline):
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    use_derivatives=use_derivatives,
                    kernel=SE(
                        use_derivatives=use_derivatives,
                        use_fingerprint=True,
                    ),
                    hpfitter=hpfitter,
                )
                # Make the fingerprint
                fp = Cartesian(
                    use_derivatives=use_derivatives,
                )
                # Set up the database
                database = Database(
                    fingerprint=fp,
                    use_derivatives=use_derivatives,
                    use_fingerprint=True,
                    round_targets=5,
                )
                # Define the machine learning model
                mlmodel = MLModel(
                    model=gp,
                    database=database,
                    optimize=True,
                    baseline=baseline,
                )
                # Construct the machine learning calculator
                mlcalc = MLCalculator(
                    mlmodel=mlmodel,
                    round_pred=5,
                )
                # Set the random seed for the calculator
                mlcalc.set_seed(seed=seed)
                # Add the training data to the calculator
                mlcalc.add_training(x_tr)
                # Test if the right number of training points is added
                self.assertTrue(mlcalc.get_training_set_size() == 10)
                # Train the machine learning calculator
                mlcalc.train_model()
                # Use a single test system for calculating the energy
                # and forces with the machine learning calculator
                atoms = x_te[0].copy()
                atoms.calc = mlcalc
                energy = atoms.get_potential_energy()
                atoms.get_forces()
                # Test the prediction energy error for a single test system
                error = abs(f_te.item(0) - energy)
                self.assertTrue(abs(error - error_list[index]) < 1e-2)


if __name__ == "__main__":
    unittest.main()
