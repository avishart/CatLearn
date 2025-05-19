import unittest
from .functions import create_func, make_train_test_set, calculate_rmse


class TestTPTrainPredict(unittest.TestCase):
    """
    Test if the Student t Process without derivatives can train and predict
    the prediction mean and variance for one and multiple test points.
    """

    def test_tp(self):
        "Test if the TP can be constructed."
        from catlearn.regression.gp.models import TProcess

        # Whether to learn from the derivatives
        use_derivatives = False
        # Construct the Studen t process
        TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )

    def test_train(self):
        "Test if the TP can be trained."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)

    def test_predict1(self):
        "Test if the TP can predict one test point."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energy
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.00069) < 1e-4)

    def test_predict(self):
        "Test if the TP can predict multiple test points."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.89152) < 1e-4)

    def test_predict_var(self):
        "Test if the TP can predict variance of multiple test point."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies and uncertainties
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=True,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.89152) < 1e-4)

    def test_predict_var_n(self):
        """
        Test if the TP can predict variance including noise
        of multiple test point.
        """
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies and uncertainties
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=True,
            get_derivatives=False,
            include_noise=True,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.89152) < 1e-4)

    def test_predict_derivatives(self):
        "Test if the TP can predict derivatives of multiple test points."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies, derivatives, and uncertainties
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=True,
            get_derivatives=True,
            include_noise=False,
        )
        # Check that the derivatives are predicted
        self.assertTrue(ypred.shape[1] == 2)
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.89152) < 1e-4)


class TestTPTrainPredictDerivatives(unittest.TestCase):
    """
    Test if the Student t Process with derivatives can train and predict
    the prediction mean and variance for one and multiple test points.
    """

    def test_train(self):
        "Test if the TP can be trained."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)

    def test_predict1(self):
        "Test if the TP can predict one test point."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energy
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.00233) < 1e-4)

    def test_predict(self):
        "Test if the TP can predict multiple test points."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.40411) < 1e-4)

    def test_predict_var(self):
        "Test if the TP can predict variance of multiple test points."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies and uncertainties
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=True,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.40411) < 1e-4)

    def test_predict_var_n(self):
        """
        Test if the TP can predict variance including noise
        of multiple test point.
        """
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies and uncertainties
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=True,
            get_derivatives=False,
            include_noise=True,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.40411) < 1e-4)

    def test_predict_derivatives(self):
        "Test if the TP can predict derivatives of multiple test points."
        from catlearn.regression.gp.models import TProcess

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
        # Construct the Studen t process
        tp = TProcess(
            hp=dict(length=[2.0], noise=[-5.0]),
            use_derivatives=use_derivatives,
        )
        # Train the machine learning model
        tp.train(x_tr, f_tr)
        # Predict the energies, derivatives, and uncertainties
        ypred, _, _ = tp.predict(
            x_te,
            get_variance=True,
            get_derivatives=True,
            include_noise=False,
        )
        # Check that the derivatives are predicted
        self.assertTrue(ypred.shape[1] == 2)
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.40411) < 1e-4)


if __name__ == "__main__":
    unittest.main()
