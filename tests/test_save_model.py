import unittest
from .functions import create_func, make_train_test_set, calculate_rmse


class TestSaveModel(unittest.TestCase):
    """
    Test if the Gaussian Process can be saved to and loaded from a pickle file.
    """

    def test_save_model(self):
        """
        Test if the Gaussian Process can be saved to and
        loaded from a pickle file.
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
        # Save the model
        gp.save_model("test_model.pkl")
        # Load the model
        gp2 = GaussianProcess(
            hp=dict(length=2.0),
            use_derivatives=use_derivatives,
        )
        gp2 = gp2.load_model("test_model.pkl")
        # Predict the energy
        ypred, _, _ = gp2.predict(
            x_te,
            get_variance=False,
            get_derivatives=False,
            include_noise=False,
        )
        # Test the prediction energy errors
        error = calculate_rmse(f_te[:, 0], ypred[:, 0])
        self.assertTrue(abs(error - 0.00859) < 1e-4)


if __name__ == "__main__":
    unittest.main()
