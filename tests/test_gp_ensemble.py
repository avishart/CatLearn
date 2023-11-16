import unittest
import numpy as np
from .functions import create_func,make_train_test_set,calculate_rmse

class TestGPEnsemble(unittest.TestCase):
    """ Test if the Ensemble of Gaussian Processes without derivatives can train and predict the prediction mean and variance. """

    def test_variance_ensemble(self):
        "Test if the the ensemble of GPs can predict multiple test points with and without variance ensemble."
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.ensemble.ensemble_clustering import EnsembleClustering
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means import K_means
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Construct the clustering object
        clustering=K_means(k=4,maxiter=20,tol=1e-3,metric='euclidean')
        # Define the list of whether to use variance as the ensemble method, which are tested
        var_list=[False,True]
        # Make a list of the error values that the test compares to
        error_list=[3.90019,1.73281]
        for index,use_variance_ensemble in enumerate(var_list):
            with self.subTest(use_variance_ensemble=use_variance_ensemble):
                # Construct the ensemble model
                enmodel=EnsembleClustering(model=gp,clustering=clustering,use_variance_ensemble=use_variance_ensemble)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Train the machine learning model
                enmodel.train(x_tr,f_tr)
                # Predict the energies 
                ypred,var=enmodel.predict(x_te,get_variance=False,get_derivatives=False,include_noise=False)
                # Test the prediction energy errors
                error=calculate_rmse(f_te[:,0],ypred[:,0])
                self.assertTrue(abs(error-error_list[index])<1e-4) 

    def test_clustering(self):
        "Test if the ensemble GPs can predict multiple test points with the different clustering algorithms."
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.ensemble.ensemble_clustering import EnsembleClustering
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means import K_means
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means_auto import K_means_auto
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means_number import K_means_number
        from catlearn.regression.gaussianprocess.ensemble.clustering.distance import DistanceClustering
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Define the list of clustering objects that are tested
        clustering_list=[K_means(k=4,maxiter=20,tol=1e-3,metric='euclidean'),
                         K_means_auto(min_data=6,max_data=12,maxiter=20,tol=1e-3,metric='euclidean'),
                         K_means_number(data_number=12,maxiter=20,tol=1e-3,metric='euclidean'),
                         DistanceClustering(centroids=np.array([[-30.0],[60.0]]),metric='euclidean')]
        # Make a list of the error values that the test compares to
        error_list=[1.73289,1.75136,1.73401,1.74409]
        # Test the baseline objects
        for index,clustering in enumerate(clustering_list):
            with self.subTest(clustering=clustering):
                # Construct the ensemble model
                enmodel=EnsembleClustering(model=gp,clustering=clustering,use_variance_ensemble=True)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Train the machine learning model
                enmodel.train(x_tr,f_tr)
                # Predict the energies and uncertainties
                ypred,var=enmodel.predict(x_te,get_variance=True,get_derivatives=False,include_noise=False)
                # Test the prediction energy errors
                error=calculate_rmse(f_te[:,0],ypred[:,0])
                self.assertTrue(abs(error-error_list[index])<1e-4) 


class TestGPEnsembleDerivatives(unittest.TestCase):
    """ Test if the Gaussian Process with derivatives can train and predict the prediction mean and variance for one and multiple test points. """  

    def test_variance_ensemble(self):
        "Test if the the ensemble of GPs can predict multiple test points with and without variance ensemble."
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.ensemble.ensemble_clustering import EnsembleClustering
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means import K_means
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Construct the clustering object
        clustering=K_means(k=4,maxiter=20,tol=1e-3,metric='euclidean')
        # Define the list of whether to use variance as the ensemble method, which are tested
        var_list=[False,True]
        # Make a list of the error values that the test compares to
        error_list=[3.66417,0.17265]
        for index,use_variance_ensemble in enumerate(var_list):
            with self.subTest(use_variance_ensemble=use_variance_ensemble):
                # Construct the ensemble model
                enmodel=EnsembleClustering(model=gp,clustering=clustering,use_variance_ensemble=use_variance_ensemble)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Train the machine learning model
                enmodel.train(x_tr,f_tr)
                # Predict the energies 
                ypred,var=enmodel.predict(x_te,get_variance=False,get_derivatives=False,include_noise=False)
                # Test the prediction energy errors
                error=calculate_rmse(f_te[:,0],ypred[:,0])
                self.assertTrue(abs(error-error_list[index])<1e-4)            

    def test_clustering(self):
        "Test if the ensemble GPs can predict multiple test points with the different clustering algorithms."
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.ensemble.ensemble_clustering import EnsembleClustering
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means import K_means
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means_auto import K_means_auto
        from catlearn.regression.gaussianprocess.ensemble.clustering.k_means_number import K_means_number
        from catlearn.regression.gaussianprocess.ensemble.clustering.distance import DistanceClustering
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Define the list of clustering objects that are tested
        clustering_list=[K_means(k=4,maxiter=20,tol=1e-3,metric='euclidean'),
                         K_means_auto(min_data=6,max_data=12,maxiter=20,tol=1e-3,metric='euclidean'),
                         K_means_number(data_number=12,maxiter=20,tol=1e-3,metric='euclidean'),
                         DistanceClustering(centroids=np.array([[-30.0],[60.0]]),metric='euclidean')]
        # Make a list of the error values that the test compares to
        error_list=[0.17265,0.15492,0.14095,0.16393]
        # Test the baseline objects
        for index,clustering in enumerate(clustering_list):
            with self.subTest(clustering=clustering):
                # Construct the ensemble model
                enmodel=EnsembleClustering(model=gp,clustering=clustering,use_variance_ensemble=True)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Train the machine learning model
                enmodel.train(x_tr,f_tr)
                # Predict the energies and uncertainties
                ypred,var=enmodel.predict(x_te,get_variance=True,get_derivatives=False,include_noise=False)
                # Test the prediction energy errors
                error=calculate_rmse(f_te[:,0],ypred[:,0])
                self.assertTrue(abs(error-error_list[index])<1e-4)

if __name__ == '__main__':
    unittest.main()

