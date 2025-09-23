import unittest
import numpy as np
from .functions import create_func, make_train_test_set, calculate_rmse


class TestGPEnsemble(unittest.TestCase):
    """
    Test if the Ensemble of Gaussian Processes without derivatives
    can train and predict the prediction mean and variance.
    """

    def test_variance_ensemble(self):
        """
        Test if the the ensemble of GPs can predict multiple test points
        with and without variance ensemble.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.ensemble import EnsembleClustering
        from catlearn.regression.gp.ensemble.clustering import K_means

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
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            use_derivatives=use_derivatives,
        )
        # Construct the clustering object
        clustering = K_means(
            n_clusters=4,
            maxiter=20,
        )
        # Define the list of whether to use variance as the ensemble method
        var_list = [False, True]
        # Make a list of the error values that the test compares to
        error_list = [4.61443, 0.48256]
        for index, use_variance_ensemble in enumerate(var_list):
            with self.subTest(use_variance_ensemble=use_variance_ensemble):
                # Construct the ensemble model
                enmodel = EnsembleClustering(
                    model=gp,
                    clustering=clustering,
                    use_variance_ensemble=use_variance_ensemble,
                )
                # Set random seed to give the same results every time
                enmodel.set_seed(seed=seed)
                # Train the machine learning model
                enmodel.train(x_tr, f_tr)
                # Predict the energies
                ypred, _, _ = enmodel.predict(
                    x_te,
                    get_variance=False,
                    get_derivatives=False,
                    include_noise=False,
                )
                # Test the prediction energy errors
                error = calculate_rmse(f_te[:, 0], ypred[:, 0])
                self.assertTrue(abs(error - error_list[index]) < 1e-4)

    def test_clustering(self):
        """
        Test if the ensemble GPs can predict multiple test points
        with the different clustering algorithms.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.ensemble import EnsembleClustering
        from catlearn.regression.gp.ensemble.clustering import (
            K_means,
            K_means_auto,
            K_means_number,
            K_means_enumeration,
            FixedClustering,
            RandomClustering,
            RandomClustering_number,
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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            use_derivatives=use_derivatives,
        )
        # Define the list of clustering objects that are tested
        clustering_list = [
            K_means(n_clusters=4, maxiter=20),
            K_means_auto(
                min_data=6,
                max_data=12,
                maxiter=20,
            ),
            K_means_number(
                data_number=12,
                maxiter=20,
            ),
            K_means_enumeration(data_number=12),
            FixedClustering(
                centroids=np.array([[-30.0], [60.0]]),
            ),
            RandomClustering(n_clusters=4, equal_size=True),
            RandomClustering_number(data_number=12),
        ]
        # Make a list of the error values that the test compares to
        error_list = [
            0.48256,
            0.63066,
            0.62649,
            0.91445,
            0.62650,
            0.70163,
            0.67975,
        ]
        # Test the baseline objects
        for index, clustering in enumerate(clustering_list):
            with self.subTest(clustering=clustering):
                # Construct the ensemble model
                enmodel = EnsembleClustering(
                    model=gp,
                    clustering=clustering,
                    use_variance_ensemble=True,
                )
                # Set random seed to give the same results every time
                enmodel.set_seed(seed=seed)
                # Train the machine learning model
                enmodel.train(x_tr, f_tr)
                # Predict the energies and uncertainties
                ypred, _, _ = enmodel.predict(
                    x_te,
                    get_variance=True,
                    get_derivatives=False,
                    include_noise=False,
                )
                # Test the prediction energy errors
                error = calculate_rmse(f_te[:, 0], ypred[:, 0])
                self.assertTrue(abs(error - error_list[index]) < 1e-4)


class TestGPEnsembleDerivatives(unittest.TestCase):
    """
    Test if the Gaussian Process with derivatives can train and predict
    the prediction mean and variance for one and multiple test points.
    """

    def test_variance_ensemble(self):
        """
        Test if the the ensemble of GPs can predict multiple test points
        with and without variance ensemble.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.ensemble import EnsembleClustering
        from catlearn.regression.gp.ensemble.clustering import K_means

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
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            use_derivatives=use_derivatives,
        )
        # Construct the clustering object
        clustering = K_means(n_clusters=4, maxiter=20)
        # Define the list of whether to use variance as the ensemble method
        var_list = [False, True]
        # Make a list of the error values that the test compares to
        error_list = [4.51161, 0.37817]
        for index, use_variance_ensemble in enumerate(var_list):
            with self.subTest(use_variance_ensemble=use_variance_ensemble):
                # Construct the ensemble model
                enmodel = EnsembleClustering(
                    model=gp,
                    clustering=clustering,
                    use_variance_ensemble=use_variance_ensemble,
                )
                # Set random seed to give the same results every time
                enmodel.set_seed(seed=seed)
                # Train the machine learning model
                enmodel.train(x_tr, f_tr)
                # Predict the energies
                ypred, _, _ = enmodel.predict(
                    x_te,
                    get_variance=False,
                    get_derivatives=False,
                    include_noise=False,
                )
                # Test the prediction energy errors
                error = calculate_rmse(f_te[:, 0], ypred[:, 0])
                self.assertTrue(abs(error - error_list[index]) < 1e-4)

    def test_clustering(self):
        """
        Test if the ensemble GPs can predict multiple test points with
        the different clustering algorithms.
        """
        from catlearn.regression.gp.models import GaussianProcess
        from catlearn.regression.gp.ensemble import EnsembleClustering
        from catlearn.regression.gp.ensemble.clustering import (
            K_means,
            K_means_auto,
            K_means_number,
            K_means_enumeration,
            FixedClustering,
            RandomClustering,
            RandomClustering_number,
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
        # Construct the Gaussian process
        gp = GaussianProcess(
            hp=dict(length=[2.0], noise=[-5.0], prefactor=[0.0]),
            use_derivatives=use_derivatives,
        )
        # Define the list of clustering objects that are tested
        clustering_list = [
            K_means(n_clusters=4, maxiter=20),
            K_means_auto(
                min_data=6,
                max_data=12,
                maxiter=20,
            ),
            K_means_number(data_number=12, maxiter=20),
            K_means_enumeration(data_number=12),
            FixedClustering(centroids=np.array([[-30.0], [60.0]])),
            RandomClustering(n_clusters=4, equal_size=True),
            RandomClustering_number(data_number=12),
        ]
        # Make a list of the error values that the test compares to
        error_list = [
            0.37817,
            0.38854,
            0.38641,
            0.52753,
            0.38640,
            0.47864,
            0.36700,
        ]
        # Test the baseline objects
        for index, clustering in enumerate(clustering_list):
            with self.subTest(clustering=clustering):
                # Construct the ensemble model
                enmodel = EnsembleClustering(
                    model=gp,
                    clustering=clustering,
                    use_variance_ensemble=True,
                )
                # Set random seed to give the same results every time
                enmodel.set_seed(seed=seed)
                # Train the machine learning model
                enmodel.train(x_tr, f_tr)
                # Predict the energies and uncertainties
                ypred, _, _ = enmodel.predict(
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
