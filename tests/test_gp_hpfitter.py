import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestGPHpfitter(unittest.TestCase):
    """ Test if the hyperparameters of the Gaussian Process can be optimized with hyperparameter fitters. """
    
    def test_hpfitters_noderiv(self):
        "Test if the hyperparameters of the GP without derivatives can be optimized."
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_optimize
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter,ReducedHyperparameterFitter,FBPMGP
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Define the list of hyperparameter fitter objects that are tested
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        hpfitter_list=[HyperparameterFitter(func=LogLikelihood(),optimization_method=local_optimize,opt_kwargs=opt_kwargs),
                       ReducedHyperparameterFitter(func=LogLikelihood(),opt_tr_size=50,optimization_method=local_optimize,opt_kwargs=opt_kwargs),
                       ReducedHyperparameterFitter(func=LogLikelihood(),opt_tr_size=10,optimization_method=local_optimize,opt_kwargs=opt_kwargs),
                       FBPMGP(Q=None,n_test=50,ngrid=80,bounds=None,hptrans=True,use_bounds=True,s=0.14,get_prior=False)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':47.042,'x':np.array([1.97,-15.39,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-15.39,1.79])},
                  {'fun':np.inf,'x':np.array([2.00,-8.00,0.00])},
                  {'fun':0.883,'x':np.array([2.00,-3.05,2.13])}]
        # Test the hyperparameter fitter objects
        for index,hpfitter in enumerate(hpfitter_list):
            with self.subTest(hpfitter=hpfitter):
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                if index!=2:
                    self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_hpfitters_deriv(self):
        "Test if the hyperparameters of the GP with derivatives can be optimized."
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_optimize
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter,ReducedHyperparameterFitter,FBPMGP
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=use_derivatives)
        # Define the list of hyperparameter fitter objects that are tested
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        hpfitter_list=[HyperparameterFitter(func=LogLikelihood(),optimization_method=local_optimize,opt_kwargs=opt_kwargs),
                       ReducedHyperparameterFitter(func=LogLikelihood(),opt_tr_size=50,optimization_method=local_optimize,opt_kwargs=opt_kwargs),
                       ReducedHyperparameterFitter(func=LogLikelihood(),opt_tr_size=10,optimization_method=local_optimize,opt_kwargs=opt_kwargs),
                       FBPMGP(Q=None,n_test=50,ngrid=80,bounds=None,hptrans=True,use_bounds=True,s=0.14,get_prior=False)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':-18.501,'x':np.array([2.00,-18.78,2.00])},
                  {'fun':-18.501,'x':np.array([2.00,-18.78,2.00])},
                  {'fun':np.inf,'x':np.array([2.00,-8.00,0.00])},
                  {'fun':-8.178,'x':np.array([1.97,-15.58,1.91])}]
        # Test the hyperparameter fitter objects
        for index,hpfitter in enumerate(hpfitter_list):
            with self.subTest(hpfitter=hpfitter):
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                if index!=2:
                    self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)


if __name__ == '__main__':
    unittest.main()

