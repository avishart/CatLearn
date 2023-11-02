import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestGPOptimizerASEParallel(unittest.TestCase):
    """ Test if the Student t process can be optimized with all existing optimization methods and objective functions that works in parallel with ASE. """

    def test_random(self):
        "Test if the TP can be local optimized from random sampling in parallel"
        try:
            from ase.parallel import world,broadcast
        except:
            self.skipTest("ASE parallel is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import RandomSamplingOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the local optimizer
        local_optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make the global optimizer
        optimizer=RandomSamplingOptimizer(local_optimizer=local_optimizer,maxiter=600,npoints=12,parallel=True)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-499.866)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.55,-2.00]))<1e-2)
   
    def test_grid(self):
        "Test if the TP can be brute-force grid optimized in parallel"
        try:
            from ase.parallel import world,broadcast
        except:
            self.skipTest("ASE parallel is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import GridOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the local optimizer
        local_optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make the global optimizer
        optimizer=GridOptimizer(local_optimizer=local_optimizer,optimize=False,maxiter=500,n_each_dim=5,parallel=True)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds_trans)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.256)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.00,-8.00]))<1e-2)

    def test_line(self):
        "Test if the TP can be iteratively line search optimized in parallel"
        try:
            from ase.parallel import world,broadcast
        except:
            self.skipTest("ASE parallel is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import IterativeLineOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the local optimizer
        local_optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make the global optimizer
        optimizer=IterativeLineOptimizer(local_optimizer=local_optimizer,optimize=False,maxiter=500,n_each_dim=10,loops=3,parallel=True)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds_trans)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.561)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.06,-70.98]))<1e-2)
    
    def test_line_search_scale(self):
        "Test if the TP can be optimized from line search in the length-scale hyperparameter in parallel"
        try:
            from ase.parallel import world,broadcast
        except:
            self.skipTest("ASE parallel is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.linesearcher import FineGridSearch
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import FactorizedOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.factorized_likelihood import FactorizedLogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        # Make the line optimizer
        line_optimizer=FineGridSearch(optimize=True,multiple_min=False,loops=3,ngrid=80,maxiter=500,jac=False,tol=1e-5,parallel=True)
        # Make the global optimizer
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=80,parallel=True)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimizer=optimizer,bounds=bounds_trans)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-499.866)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.55,-2.00]))<1e-2)

if __name__ == '__main__':
    unittest.main()
