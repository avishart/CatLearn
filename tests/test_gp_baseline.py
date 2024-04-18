import unittest
import numpy as np
from .functions import create_h2_atoms,make_train_test_set

class TestGPBaseline(unittest.TestCase):
    """ Test if the baseline can be used in the Gaussian process as an ASE calculator with different database forms for ASE atoms. """

    def test_predict(self):
        "Test if the GP calculator can predict energy and forces"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.kernel import SE
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.fingerprint import Cartesian
        from catlearn.regression.gaussianprocess.calculator import Database,MLModel,MLCalculator
        from catlearn.regression.gaussianprocess.baseline.baseline import Baseline_calculator
        from catlearn.regression.gaussianprocess.baseline.repulsive import Repulsion_calculator
        from catlearn.regression.gaussianprocess.baseline.mie import Mie_calculator
        # Create the data set
        x,f,g=create_h2_atoms(gridsize=50,seed=1)
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=10,te=1,use_derivatives=use_derivatives)
        # Make the hyperparameter fitter
        optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-8)
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Define the list of baseline objects that are tested
        baseline_list=[Baseline_calculator(),Repulsion_calculator(r_scale=0.7),Mie_calculator()]
        # Make a list of the error values that the test compares to
        error_list=[0.00165,1.93820,3.33650]
        # Test the baseline objects
        for index,baseline in enumerate(baseline_list):
            with self.subTest(baseline=baseline):
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),
                                   use_derivatives=use_derivatives,
                                   kernel=SE(use_derivatives=use_derivatives,use_fingerprint=True),
                                   hpfitter=hpfitter)
                # Make the fingerprint
                fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True)
                # Set up the database
                database=Database(fingerprint=fp,
                                  reduce_dimensions=True,
                                  use_derivatives=use_derivatives,
                                  negative_forces=True,
                                  use_fingerprint=True)
                # Define the machine learning model
                mlmodel=MLModel(model=gp,database=database,optimize=True,baseline=baseline)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Construct the machine learning calculator and add the data
                mlcalc=MLCalculator(mlmodel=mlmodel,calculate_uncertainty=True,calculate_forces=True,verbose=False)
                mlcalc.add_training(x_tr)
                # Test if the right number of training points is added
                self.assertTrue(mlcalc.get_training_set_size()==10)
                # Train the machine learning calculator
                mlcalc.train_model()
                # Use a single test system for calculating the energy and forces with the machine learning calculator
                atoms=x_te[0].copy()
                atoms.calc=mlcalc
                energy=atoms.get_potential_energy()
                atoms.get_forces()
                # Test the prediction energy error for a single test system
                error=abs(f_te.item(0)-energy)
                self.assertTrue(abs(error-error_list[index])<1e-4)

if __name__ == '__main__':
    unittest.main()

