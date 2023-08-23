import unittest
import numpy as np
from .functions import create_func,create_h2_atoms,make_train_test_set,calculate_rmse

class TestGPFP(unittest.TestCase):
    """ Test if the Gaussian Process without derivatives can train and predict with fingerprints. """

    def test_predict_var(self):
        "Test if the GP can predict variance of multiple test point with fingerprints"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.kernel import SE
        from catlearn.regression.gaussianprocess.fingerprint import Cartesian,Coulomb,Inv_distances,Sum_distances,Sum_distances_power,Mean_distances,Mean_distances_power
        # Create the data set
        x,f,g=create_h2_atoms(gridsize=50,seed=1)
        ## Whether to learn from the derivatives
        use_derivatives=False
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives,kernel=SE(use_derivatives=use_derivatives,use_fingerprint=True))
        # Define the list of fingerprint objects that are tested
        fp_kwarg_list=[Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Coulomb(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Inv_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True,sorting=True),
                       Inv_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True,sorting=False),
                       Sum_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Sum_distances_power(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Mean_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Mean_distances_power(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True)]
        # Make a list of the error values that the test compares to
        error_list=[1.35605,0.52706,0.65313,0.65313,0.65313,0.45222,0.65313,0.45222]
        # Test the fingerprint objects
        for index,fp in enumerate(fp_kwarg_list):
            with self.subTest(fp=fp):
                # Construct the fingerprints
                fps=[fp(xi) for xi in x]
                x_tr,f_tr,x_te,f_te=make_train_test_set(fps,f,g,tr=10,te=10,use_derivatives=use_derivatives)
                # Train the machine learning model
                gp.train(x_tr,f_tr)
                # Predict the energies and uncertainties
                ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=False,include_noise=False)
                # Test the prediction energy errors
                error=calculate_rmse(f_te[:,0],ypred[:,0])
                self.assertTrue(abs(error-error_list[index])<1e-4)


class TestGPFPDerivatives(unittest.TestCase):
    """ Test if the Gaussian Process with derivatives can train and predict with fingerprints. """

    def test_predict_var(self):
        "Test if the GP can predict variance of multiple test point with fingerprints"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.kernel import SE
        from catlearn.regression.gaussianprocess.fingerprint import Cartesian,Coulomb,Inv_distances,Sum_distances,Sum_distances_power,Mean_distances,Mean_distances_power
        # Create the data set
        x,f,g=create_h2_atoms(gridsize=50,seed=1)
        ## Whether to True from the derivatives
        use_derivatives=True
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),use_derivatives=use_derivatives,kernel=SE(use_derivatives=use_derivatives,use_fingerprint=True))
        # Define the list of fingerprint objects that are tested
        fp_kwarg_list=[Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Coulomb(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Inv_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True,sorting=True),
                       Inv_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True,sorting=False),
                       Sum_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Sum_distances_power(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Mean_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True),
                       Mean_distances_power(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True)]
        # Make a list of the error values that the test compares to
        error_list=[22.75648,1.30271,9.90152,9.90152,9.90152,8.60277,9.90152,8.60277]
        # Test the fingerprint objects
        for index,fp in enumerate(fp_kwarg_list):
            with self.subTest(fp=fp):
                # Construct the fingerprints
                fps=[fp(xi) for xi in x]
                x_tr,f_tr,x_te,f_te=make_train_test_set(fps,f,g,tr=10,te=10,use_derivatives=use_derivatives)
                # Train the machine learning model
                gp.train(x_tr,f_tr)
                # Predict the energies and uncertainties
                ypred,var=gp.predict(x_te,get_variance=True,get_derivatives=False,include_noise=False)
                # Test the prediction energy errors
                error=calculate_rmse(f_te[:,0],ypred[:,0])
                self.assertTrue(abs(error-error_list[index])<1e-4)


if __name__ == '__main__':
    unittest.main()

