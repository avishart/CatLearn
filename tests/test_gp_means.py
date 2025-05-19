import unittest
from .functions import create_func, make_train_test_set, calculate_rmse


class TestGPMeans(unittest.TestCase):
    """
    Test if the Gaussian Process works with different prior means.
    """

    def test_means_noderiv(self):
        """
        Test if the GP without derivatives can train and predict
        multiple test points with different prior means.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.means import (
            Prior_constant,
            Prior_mean,
            Prior_median,
            Prior_min,
            Prior_max,
            Prior_first,
        )

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
        # Define the list of prior mean objects that are tested
        priors = [
            Prior_constant,
            Prior_mean,
            Prior_median,
            Prior_min,
            Prior_max,
            Prior_first,
        ]
        # Make a list of the error values that the test compares to
        error_list = [2.61859, 0.89152, 0.91990, 1.21032, 0.61772, 0.91545]
        # Test the prior mean objects
        for index, prior in enumerate(priors):
            with self.subTest(prior=prior):
                # Construct the Gaussian process
                gp = GaussianProcess(
                    prior=prior(),
                    hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                gp.set_seed(seed)
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

    def test_means_deriv(self):
        """
        Test if the GP with derivatives can train and predict
        multiple test points with different prior means.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.means import (
            Prior_constant,
            Prior_mean,
            Prior_median,
            Prior_min,
            Prior_max,
            Prior_first,
        )

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
        # Define the list of prior mean objects that are tested
        priors = [
            Prior_constant,
            Prior_mean,
            Prior_median,
            Prior_min,
            Prior_max,
            Prior_first,
        ]
        # Make a list of the error values that the test compares to
        error_list = [1.14773, 0.40411, 0.41732, 0.54772, 0.26334, 0.41526]
        # Test the prior mean objects
        for index, prior in enumerate(priors):
            with self.subTest(prior=prior):
                # Construct the Gaussian process
                gp = GaussianProcess(
                    prior=prior(),
                    hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
                    use_derivatives=use_derivatives,
                )
                # Set random seed to give the same results every time
                gp.set_seed(seed)
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
