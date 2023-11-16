import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestGPObjectiveFunctions(unittest.TestCase):
    """ Test if the Gaussian Process can be optimized with all existing objective functions. """
    
    def test_local(self):
        "Test if the GP can be local optimized with all objective functions that does not use eigendecomposition"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.objectivefunctions.gp import LogLikelihood,MaximumLogLikelihood,GPP,LOO,GPE
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Define the list of objective function objects that are tested
        obj_list=[LogLikelihood(),
                  MaximumLogLikelihood(modification=False),
                  MaximumLogLikelihood(modification=True),
                  GPP(),
                  LOO(use_analytic_prefactor=False),
                  LOO(use_analytic_prefactor=True),
                  GPE()]
        # Make the optimizer
        optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':47.042,'x':np.array([1.97,-15.39,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-17.48,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-17.48,1.84])},
                  {'fun':0.803, 'x':np.array([2.91,-25.69,8.22])},
                  {'fun':7.098, 'x':np.array([1.70,-12.07,0.00])},
                  {'fun':7.098, 'x':np.array([1.70,-12.07,1.53])},
                  {'fun':7.098,'x':np.array([1.70,-15.72,-25.70])}]
        # Test the objective function objects
        for index,obj_func in enumerate(obj_list):
            with self.subTest(obj_func=obj_func):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=obj_func,optimizer=optimizer)
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)
    
    def test_line_search_scale(self):
        "Test if the GP can be optimized from line search in the length-scale hyperparameter with all objective functions that uses eigendecomposition"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import FactorizedOptimizer
        from catlearn.regression.gaussianprocess.optimizers.linesearcher import FineGridSearch
        from catlearn.regression.gaussianprocess.optimizers.noisesearcher import NoiseGrid,NoiseGoldenSearch,NoiseFineGridSearch
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.objectivefunctions.gp import FactorizedLogLikelihood,FactorizedLogLikelihoodSVD,FactorizedGPP
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make fixed boundary conditions for one of the tests
        fixed_bounds=HPBoundaries(bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]]),log=True)
        # Make the optimizers
        line_optimizer=FineGridSearch(tol=1e-5,loops=3,ngrid=80,optimize=True,multiple_min=True)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,maxiter=500,ngrid=80,parallel=False)
        # Define the list of objective function objects that are tested
        obj_list=[(None,FactorizedLogLikelihood(modification=False,ngrid=250,noise_optimizer=NoiseGrid())),
                  (None,FactorizedLogLikelihood(modification=True,ngrid=250,noise_optimizer=NoiseGrid())),
                  (None,FactorizedLogLikelihood(modification=False,ngrid=80,noise_optimizer=NoiseGoldenSearch())),
                  (None,FactorizedLogLikelihood(modification=False,ngrid=80,noise_optimizer=NoiseFineGridSearch())),
                  (fixed_bounds,FactorizedLogLikelihood(modification=False,ngrid=250,noise_optimizer=NoiseGrid())),
                  (None,FactorizedLogLikelihoodSVD(modification=False,ngrid=250,noise_optimizer=NoiseGrid())),
                  (None,FactorizedGPP(modification=False,ngrid=250,noise_optimizer=NoiseGrid()))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':44.703,'x':np.array([2.55,-2.01,1.70])},
                  {'fun':44.703,'x':np.array([2.55,-2.01,1.72])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.703,'x':np.array([2.55,-2.01,1.70])},
                  {'fun':-2.843,'x':np.array([2.96,-70.98,6.82])}]
        # Test the objective function objects
        for index,(bounds,obj_func) in enumerate(obj_list):
            with self.subTest(bounds=bounds,obj_func=obj_func):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=obj_func,optimizer=optimizer,bounds=bounds)
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)


if __name__ == '__main__':
    unittest.main()

