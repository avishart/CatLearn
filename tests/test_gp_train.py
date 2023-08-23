import unittest
import numpy as np
from .functions import create_func,make_train_test_set,calculate_rmse

class TestGPTrainPredict(unittest.TestCase):
    """ Test if the Gaussian Process without derivatives can train and predict the prediction mean and variance for one and multiple test points. """

    def test_gp(self):
        " Test if the GP can be constructed "
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Whether to learn from the derivatives
        use_derivatives=False
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)

    def test_train(self):
        "Test if the GP can be trained"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)

    def test_predict1(self):
        "Test if the GP can predict one test point"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energy
        ypred,var=gp.predict(x_te,get_variance=False,get_derivatives=False,include_noise=False)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-0.02650)<1e-4) 

    def test_predict(self):
        "Test if the GP can predict multiple test points"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies
        ypred,var=gp.predict(x_te,get_variance=False,get_derivatives=False,include_noise=False)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-1.75102)<1e-4) 

    def test_predict_var(self):
        "Test if the GP can predict variance of multiple test point"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies and uncertainties
        ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=False,include_noise=False)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-1.75102)<1e-4) 

    def test_predict_var_n(self):
        "Test if the GP can predict variance including noise of multiple test point"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies and uncertainties
        ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=False,include_noise=True)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-1.75102)<1e-4)

    def test_predict_derivatives(self):
        "Test if the GP can predict derivatives of multiple test points"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies, derivatives, and uncertainties
        ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=True,include_noise=False)
        # Check that the derivatives are predicted
        self.assertTrue(np.shape(ypred)[1]==2)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-1.75102)<1e-4)


class TestGPTrainPredictDerivatives(unittest.TestCase):
    """ Test if the Gaussian Process with derivatives can train and predict the prediction mean and variance for one and multiple test points. """

    def test_train(self):
        "Test if the GP can be trained"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)

    def test_predict1(self):
        "Test if the GP can predict one test point"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energy
        ypred,var=gp.predict(x_te,get_variance=False,get_derivatives=False,include_noise=False)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-0.00218)<1e-4) 

    def test_predict(self):
        "Test if the GP can predict multiple test points"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies
        ypred,var=gp.predict(x_te,get_variance=False,get_derivatives=False,include_noise=False)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-0.13723)<1e-4) 

    def test_predict_var(self):
        "Test if the GP can predict variance of multiple test points"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies and uncertainties
        ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=False,include_noise=False)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-0.13723)<1e-4) 

    def test_predict_var_n(self):
        "Test if the GP can predict variance including noise of multiple test point"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies and uncertainties
        ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=False,include_noise=True)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-0.13723)<1e-4)

    def test_predict_derivatives(self):
        "Test if the GP can predict derivatives of multiple test points"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives)
        # Train the machine learning model
        gp.train(x_tr,f_tr)
        # Predict the energies, derivatives, and uncertainties
        ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=True,include_noise=False)
        # Check that the derivatives are predicted
        self.assertTrue(np.shape(ypred)[1]==2)
        # Test the prediction energy errors
        error=calculate_rmse(f_te[:,0],ypred[:,0])
        self.assertTrue(abs(error-0.13723)<1e-4)

if __name__ == '__main__':
    unittest.main()

