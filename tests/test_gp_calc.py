import unittest
from .functions import create_h2_atoms, make_train_test_set


class TestGPCalc(unittest.TestCase):
    """
    Test if the Gaussian Process can be used as
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
            DatabaseDistance,
            DatabaseHybrid,
            DatabaseMin,
            DatabaseRandom,
            DatabaseLast,
            DatabaseRestart,
            DatabasePointsInterest,
            DatabasePointsInterestEach,
        )
        from catlearn.regression.gp.calculator import MLModel, MLCalculator

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
            jac=True,
        )
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
            round_hp=3,
        )
        # Set the maximum number of points to use for the reduced databases
        npoints = 8
        # Define the list of database objects that are tested
        data_kwargs = [
            (Database, False, dict()),
            (Database, True, dict()),
            (
                DatabaseDistance,
                True,
                dict(npoints=npoints, initial_indicies=[0]),
            ),
            (
                DatabaseDistance,
                True,
                dict(npoints=npoints, initial_indicies=[]),
            ),
            (
                DatabaseHybrid,
                True,
                dict(npoints=npoints, initial_indicies=[0]),
            ),
            (DatabaseHybrid, True, dict(npoints=npoints, initial_indicies=[])),
            (DatabaseMin, True, dict(npoints=npoints, initial_indicies=[0])),
            (DatabaseMin, True, dict(npoints=npoints, initial_indicies=[])),
            (
                DatabaseRandom,
                True,
                dict(npoints=npoints, initial_indicies=[0]),
            ),
            (DatabaseRandom, True, dict(npoints=npoints, initial_indicies=[])),
            (DatabaseLast, True, dict(npoints=npoints, initial_indicies=[0])),
            (DatabaseLast, True, dict(npoints=npoints, initial_indicies=[])),
            (
                DatabaseRestart,
                True,
                dict(npoints=npoints, initial_indicies=[0]),
            ),
            (
                DatabaseRestart,
                True,
                dict(npoints=npoints, initial_indicies=[]),
            ),
            (
                DatabasePointsInterest,
                True,
                dict(
                    npoints=npoints, initial_indicies=[0], point_interest=x_te
                ),
            ),
            (
                DatabasePointsInterest,
                True,
                dict(
                    npoints=npoints, initial_indicies=[], point_interest=x_te
                ),
            ),
            (
                DatabasePointsInterestEach,
                True,
                dict(
                    npoints=npoints, initial_indicies=[0], point_interest=x_te
                ),
            ),
            (
                DatabasePointsInterestEach,
                True,
                dict(
                    npoints=npoints, initial_indicies=[], point_interest=x_te
                ),
            ),
        ]
        # Make a list of the error values that the test compares to
        error_list = [
            2.11773,
            2.11773,
            0.33617,
            0.33617,
            1.95853,
            0.33617,
            0.71664,
            0.71664,
            0.89497,
            0.35126,
            5.04806,
            6.25093,
            7.38153,
            9.47098,
            1.76828,
            1.76828,
            1.76828,
            1.76828,
        ]
        # Test the database objects
        for index, (data, use_fingerprint, data_kwarg) in enumerate(
            data_kwargs
        ):
            with self.subTest(
                data=data,
                use_fingerprint=use_fingerprint,
                data_kwarg=data_kwarg,
            ):
                # Construct the Gaussian process
                gp = GaussianProcess(
                    hp=dict(length=2.0),
                    use_derivatives=use_derivatives,
                    kernel=SE(
                        use_derivatives=use_derivatives,
                        use_fingerprint=use_fingerprint,
                    ),
                    hpfitter=hpfitter,
                )
                # Make the fingerprint
                fp = Cartesian(
                    use_derivatives=use_derivatives,
                )
                # Set up the database
                database = data(
                    fingerprint=fp,
                    use_derivatives=use_derivatives,
                    use_fingerprint=use_fingerprint,
                    round_targets=5,
                    **data_kwarg
                )
                # Define the machine learning model
                mlmodel = MLModel(
                    model=gp,
                    database=database,
                    optimize=True,
                    baseline=None,
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
                if index in [0, 1]:
                    self.assertTrue(
                        len(mlcalc.mlmodel.database.get_features()) == 10
                    )
                elif index == 12:
                    self.assertTrue(
                        len(mlcalc.mlmodel.database.get_features()) == 3
                    )
                elif index == 13:
                    self.assertTrue(
                        len(mlcalc.mlmodel.database.get_features()) == 2
                    )
                else:
                    self.assertTrue(
                        len(mlcalc.mlmodel.database.get_features()) == npoints
                    )
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

    def test_bayesian_calc(self):
        "Test if the GP bayesian calculator can predict energy and forces."
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.kernel import SE
        from catlearn.regression.gp.optimizers import ScipyOptimizer
        from catlearn.regression.gp.objectivefunctions.gp import LogLikelihood
        from catlearn.regression.gp.hpfitter import HyperparameterFitter
        from catlearn.regression.gp.fingerprint import Cartesian
        from catlearn.regression.gp.calculator import Database
        from catlearn.regression.gp.calculator import MLModel, BOCalculator

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
            jac=True,
        )
        hpfitter = HyperparameterFitter(
            func=LogLikelihood(),
            optimizer=optimizer,
            round_hp=3,
        )
        # Make the fingerprint
        use_fingerprint = True
        fp = Cartesian(
            use_derivatives=use_derivatives,
        )
        # Set up the database
        database = Database(
            fingerprint=fp,
            use_derivatives=use_derivatives,
            use_fingerprint=use_fingerprint,
            round_targets=5,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
            kernel=SE(
                use_derivatives=use_derivatives,
                use_fingerprint=use_fingerprint,
            ),
            hpfitter=hpfitter,
        )
        # Define the machine learning model
        mlmodel = MLModel(
            model=gp,
            database=database,
            optimize=True,
            baseline=None,
        )
        # Construct the machine learning calculator
        mlcalc = BOCalculator(
            mlmodel=mlmodel,
            kappa=2.0,
            round_pred=5,
        )
        # Set the random seed for the calculator
        mlcalc.set_seed(seed=seed)
        # Add the training data to the calculator
        mlcalc.add_training(x_tr)
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
        self.assertTrue(abs(error - 1.32997) < 1e-2)


if __name__ == "__main__":
    unittest.main()
