import unittest
from .functions import create_h2_atoms, make_train_test_set, calculate_rmse


class TestGPFP(unittest.TestCase):
    """
    Test if the Gaussian Process without derivatives can
    train and predict with fingerprints.
    """

    def test_predict_var(self):
        """
        Test if the GP can predict variance of multiple test point
        with fingerprints.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.kernel import SE
        from catlearn.regression.gp.fingerprint import (
            Cartesian,
            Distances,
            InvDistances,
            InvDistances2,
            SortedInvDistances,
            SumDistances,
            SumDistancesPower,
            MeanDistances,
            MeanDistancesPower,
        )

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_h2_atoms(gridsize=50, seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = False
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            use_derivatives=use_derivatives,
            kernel=SE(use_derivatives=use_derivatives, use_fingerprint=True),
        )
        # Define the list of fingerprint objects that are tested
        fp_kwarg_list = [
            Cartesian(reduce_dimensions=True, use_derivatives=use_derivatives),
            Distances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            InvDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            InvDistances2(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            SortedInvDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            SumDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            SumDistancesPower(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
                power=4,
            ),
            MeanDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            MeanDistancesPower(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=False,
                power=4,
            ),
        ]
        # Make a list of the error values that the test compares to
        error_list = [
            23.51556,
            22.50691,
            10.00542,
            56.04324,
            10.00542,
            6.712740,
            13.49250,
            20.04389,
            1.880300,
        ]
        # Test the fingerprint objects
        for index, fp in enumerate(fp_kwarg_list):
            with self.subTest(fp=fp):
                # Construct the fingerprints
                fps = [fp(xi) for xi in x]
                x_tr, f_tr, x_te, f_te = make_train_test_set(
                    fps,
                    f,
                    g,
                    tr=10,
                    te=10,
                    use_derivatives=use_derivatives,
                )
                # Set the random seed
                gp.set_seed(seed=seed)
                # Train the machine learning model
                gp.train(x_tr, f_tr)
                # Predict the energies and uncertainties
                ypred, _, _ = gp.predict(
                    x_te,
                    get_variance=True,
                    get_derivatives=False,
                    include_noise=False,
                )
                # Test the prediction energy errors
                error = calculate_rmse(f_te[:, 0], ypred[:, 0])
                self.assertTrue(abs(error - error_list[index]) < 1e-4)


class TestGPFPDerivatives(unittest.TestCase):
    """
    Test if the Gaussian Process with derivatives can train and predict
    with fingerprints.
    """

    def test_predict_var(self):
        """
        Test if the GP can predict variance of multiple test point
        with fingerprints
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.kernel import SE
        from catlearn.regression.gp.fingerprint import (
            Cartesian,
            Distances,
            InvDistances,
            InvDistances2,
            SortedInvDistances,
            SumDistances,
            SumDistancesPower,
            MeanDistances,
            MeanDistancesPower,
        )

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_h2_atoms(gridsize=50, seed=seed)
        # Whether to True from the derivatives
        use_derivatives = True
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            use_derivatives=use_derivatives,
            kernel=SE(use_derivatives=use_derivatives, use_fingerprint=True),
        )
        # Define the list of fingerprint objects that are tested
        fp_kwarg_list = [
            Cartesian(reduce_dimensions=True, use_derivatives=use_derivatives),
            Distances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            InvDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            InvDistances2(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            SortedInvDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            SumDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            SumDistancesPower(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
                power=4,
            ),
            MeanDistances(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=True,
            ),
            MeanDistancesPower(
                reduce_dimensions=True,
                use_derivatives=use_derivatives,
                periodic_softmax=False,
                power=4,
            ),
        ]
        # Make a list of the error values that the test compares to
        error_list = [
            37.64770,
            39.70638,
            69.16602,
            58.86160,
            69.16602,
            73.85387,
            69.11083,
            63.00867,
            70.55665,
        ]
        # Test the fingerprint objects
        for index, fp in enumerate(fp_kwarg_list):
            with self.subTest(fp=fp):
                # Construct the fingerprints
                fps = [fp(xi) for xi in x]
                x_tr, f_tr, x_te, f_te = make_train_test_set(
                    fps,
                    f,
                    g,
                    tr=10,
                    te=10,
                    use_derivatives=use_derivatives,
                )
                # Set random seed
                gp.set_seed(seed=seed)
                # Train the machine learning model
                gp.train(x_tr, f_tr)
                # Predict the energies and uncertainties
                ypred, _, _ = gp.predict(
                    x_te,
                    get_variance=True,
                    get_derivatives=False,
                    include_noise=False,
                )
                # Test the prediction energy errors
                error = calculate_rmse(f_te[:, 0], ypred[:, 0])
                self.assertTrue(abs(error - error_list[index]) < 1e-4)


if __name__ == "__main__":
    unittest.main()
