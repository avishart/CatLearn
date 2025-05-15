import unittest
from .functions import create_func, make_train_test_set, calculate_rmse


class TestGPTrainPredict(unittest.TestCase):
    """
    Test if the Gaussian Process without derivatives can train and
    predict the prediction mean and variance for one and multiple test points.
    """

    def test_gp(self):
        "Test if the GP can be constructed."
        from catlearn.regression.gp.models import GaussianProcess

        # Whether to learn from the derivatives
        use_derivatives = False
        # Construct the Gaussian process
        GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )

    def test_train(self):
        "Test if the GP can be trained."
        from catlearn.regression.gp.models import GaussianProcess

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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)

    def test_predict1(self):
        "Test if the GP can predict one test point."
        from catlearn.regression.gp.models import GaussianProcess

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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energy
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.00859) < 1e-4)

    def test_predict(self):
        "Test if the GP can predict multiple test points."
        from catlearn.regression.gp.models import GaussianProcess

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
            te=20,
            use_derivatives=use_derivatives,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energies
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.88457) < 1e-4)

    def test_predict_var(self):
        "Test if the GP can predict variance of multiple test point."
        from catlearn.regression.gp.models import GaussianProcess

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
            te=20,
            use_derivatives=use_derivatives,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
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
        self.assertTrue(abs(error - 0.88457) < 1e-4)

    def test_predict_var_n(self):
        """
        Test if the GP can predict variance including noise
        of multiple test point.
        """
        from catlearn.regression.gp.models import GaussianProcess

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
            te=20,
            use_derivatives=use_derivatives,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energies and uncertainties
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=True,
            get_derivatives=False,
            include_noise=True,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.88457) < 1e-4)

    def test_predict_derivatives(self):
        "Test if the GP can predict derivatives of multiple test points."
        from catlearn.regression.gp.models import GaussianProcess

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
            te=20,
            use_derivatives=use_derivatives,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energies, derivatives, and uncertainties
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=True,
            get_derivatives=True,
            include_noise=False,
        )
        # Check that the derivatives are predicted
        self.assertTrue(ypred.shape[1] == 2)
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.88457) < 1e-4)


class TestGPTrainPredictDerivatives(unittest.TestCase):
    """
    Test if the Gaussian Process with derivatives can train and predict
    the prediction mean and variance for one and multiple test points.
    """

    def test_train(self):
        "Test if the GP can be trained."
        from catlearn.regression.gp.models import GaussianProcess

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = True
        x_tr, f_tr, _, _ = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)

    def test_predict1(self):
        "Test if the GP can predict one test point."
        from catlearn.regression.gp.models import GaussianProcess

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
        # Whether to learn from the derivatives
        use_derivatives = True
        x_tr, f_tr, x_te, f_te = make_train_test_set(
            x,
            f,
            g,
            tr=20,
            te=1,
            use_derivatives=use_derivatives,
        )
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energy
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.00038) < 1e-4)

    def test_predict(self):
        "Test if the GP can predict multiple test points."
        from catlearn.regression.gp.models import GaussianProcess

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energies
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.2055) < 1e-4)

    def test_predict_var(self):
        "Test if the GP can predict variance of multiple test points."
        from catlearn.regression.gp.models import GaussianProcess

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
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
        self.assertTrue(abs(error - 0.20550) < 1e-4)

    def test_predict_var_n(self):
        """
        Test if the GP can predict variance including noise
        of multiple test point.
        """
        from catlearn.regression.gp.models import GaussianProcess

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energies and uncertainties
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=True,
            get_derivatives=False,
            include_noise=True,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.20550) < 1e-4)

    def test_predict_derivatives(self):
        "Test if the GP can predict derivatives of multiple test points."
        from catlearn.regression.gp.models import GaussianProcess

        # Set random seed to give the same results every time
        seed = 1
        # Create the data set
        x, f, g = create_func(seed=seed)
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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        gp.train(x_tr, f_tr)
        # Predict the energies, derivatives, and uncertainties
        ypred, _, _ = gp.predict(
            x_te,
            get_variance=True,
            get_derivatives=True,
            include_noise=False,
        )
        # Check that the derivatives are predicted
        self.assertTrue(ypred.shape[1] == 2)
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.20550) < 1e-4)


if __name__ == "__main__":
    unittest.main()
