import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestTPOptimizerParallel(unittest.TestCase):
    """ Test if the Student T Process can be optimized with all existing optimization methods and objective functions that works in parallel. """
    
    def test_random(self):
        "Test if the TP can be local optimized from random sampling in parallel"
        try:
            from mpi4py import MPI
        except:
            self.skipTest("MPI4PY is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.mpi_global_opt import random_parallel
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(local_run=scipy_opt,maxiter=600,jac=True,npoints=12,bounds=None,hptrans=True,use_bounds=True,local_kwargs=local_kwargs)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=random_parallel,opt_kwargs=opt_kwargs)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-499.866)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.55,-2.00]))<1e-2)
    
    def test_grid(self):
        "Test if the TP can be brute-force grid optimized in parallel"
        try:
            from mpi4py import MPI
        except:
            self.skipTest("MPI4PY is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.mpi_global_opt import grid_parallel
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,bounds=None,n_each_dim=5,hptrans=True,use_bounds=True,local_kwargs=dict())
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=grid_parallel,opt_kwargs=opt_kwargs)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.256)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.00,-8.00]))<1e-2)
    
    def test_line_search_scale(self):
        "Test if the TP can be optimized from line search in the length-scale hyperparameter in parallel "
        try:
            from mpi4py import MPI
        except:
            self.skipTest("MPI4PY is not installed")
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import fine_grid_search
        from catlearn.regression.gaussianprocess.optimizers.mpi_global_opt import line_search_scale_parallel,calculate_list_values_parallelize
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.factorized_likelihood import FactorizedLogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(fun_list=calculate_list_values_parallelize,tol=1e-5,loops=3,iterloop=80,optimize=False,multiple_min=False)
        opt_kwargs=dict(local_run=fine_grid_search,maxiter=500,jac=False,ngrid=80,bounds=None,hptrans=True,use_bounds=True,local_kwargs=local_kwargs)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimization_method=line_search_scale_parallel,opt_kwargs=opt_kwargs)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-499.866)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.55,-2.00]))<1e-2)

if __name__ == '__main__':
    unittest.main()
