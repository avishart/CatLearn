import unittest
import numpy as np
from .functions import create_func,create_h2_atoms,make_train_test_set

class TestGPCalc(unittest.TestCase):
    """ Test if the Gaussian Process can be used as an ASE calculator with different database forms for ASE atoms. """

    def test_predict(self):
        "Test if the GP calculator can predict energy and forces"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.kernel import SE
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.fingerprint import Cartesian
        from catlearn.regression.gaussianprocess.calculator import Database,DatabaseDistance,DatabaseHybrid,DatabaseMin,DatabaseRandom,DatabaseLast,DatabaseRestart,DatabasePointsInterest,MLModel,MLCalculator
        # Create the data set
        x,f,g=create_h2_atoms(gridsize=50,seed=1)
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=10,te=1,use_derivatives=use_derivatives)
        # Make the hyperparameter fitter
        optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-8)
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Set the maximum number of points to use for the reduced databases
        npoints=8
        # Define the list of database objects that are tested
        data_kwargs=[(Database,False,dict()),
                     (Database,True,dict()),
                     (DatabaseDistance,True,dict(npoints=npoints,initial_indicies=[0])),
                     (DatabaseDistance,True,dict(npoints=npoints,initial_indicies=[])),
                     (DatabaseHybrid,True,dict(npoints=npoints,initial_indicies=[0])),
                     (DatabaseHybrid,True,dict(npoints=npoints,initial_indicies=[])),
                     (DatabaseMin,True,dict(npoints=npoints,initial_indicies=[0])),
                     (DatabaseMin,True,dict(npoints=npoints,initial_indicies=[])),
                     (DatabaseRandom,True,dict(npoints=npoints,initial_indicies=[0])),
                     (DatabaseRandom,True,dict(npoints=npoints,initial_indicies=[])),
                     (DatabaseLast,True,dict(npoints=npoints,initial_indicies=[0])),
                     (DatabaseLast,True,dict(npoints=npoints,initial_indicies=[])),
                     (DatabaseRestart,True,dict(npoints=npoints,initial_indicies=[0])),
                     (DatabaseRestart,True,dict(npoints=npoints,initial_indicies=[])),
                     (DatabasePointsInterest,True,dict(npoints=npoints,initial_indicies=[0],point_interest=x_te)),
                     (DatabasePointsInterest,True,dict(npoints=npoints,initial_indicies=[],point_interest=x_te))]
        # Make a list of the error values that the test compares to
        error_list=[0.00166,0.00166,0.00359,0.00359,0.00003,0.00003,0.000002,0.000002,0.000018,0.00003,0.01270,0.02064,0.00655,0.00102,0.000002,0.000002]
        # Test the database objects
        for index,(data,use_fingerprint,data_kwarg) in enumerate(data_kwargs):
            with self.subTest(data=data,use_fingerprint=use_fingerprint,data_kwarg=data_kwarg):
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),
                                   use_derivatives=use_derivatives,
                                   kernel=SE(use_derivatives=use_derivatives,use_fingerprint=use_fingerprint),
                                   hpfitter=hpfitter)
                # Make the fingerprint
                fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True)
                # Set up the database
                database=data(fingerprint=fp,
                              reduce_dimensions=True,
                              use_derivatives=use_derivatives,
                              negative_forces=True,
                              use_fingerprint=use_fingerprint,**data_kwarg)
                # Define the machine learning model
                mlmodel=MLModel(model=gp,database=database,optimize=True,baseline=None,verbose=False)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Construct the machine learning calculator and add the data
                mlcalc=MLCalculator(mlmodel=mlmodel,calculate_uncertainty=True,calculate_forces=True)
                mlcalc.add_training(x_tr)
                # Test if the right number of training points is added
                if index in [0,1]:
                    self.assertTrue(len(mlcalc.mlmodel.database.get_features())==10)
                elif index==12:
                    self.assertTrue(len(mlcalc.mlmodel.database.get_features())==3)
                elif index==13:
                    self.assertTrue(len(mlcalc.mlmodel.database.get_features())==2)
                else:
                    self.assertTrue(len(mlcalc.mlmodel.database.get_features())==npoints)
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

