import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestGPOptimizer(unittest.TestCase):
    """ Test if the Gaussian Process can be optimized with all existing optimization methods and objective functions. """

    def test_function(self):
        "Test if the function value of the LogLikelihood is obtainable "
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.global_opt import function
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=function)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-393.422)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.00,-8.00,0.00]))<1e-2)
    
    def test_local_jac(self):
        "Test if the GP can be local optimized with gradients wrt the hyperparameters"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_optimize
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=local_optimize,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-47.042)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-15.39,1.79]))<1e-2)

    def test_local_nojac(self):
        "Test if the GP can be local optimized without gradients wrt the hyperparameters"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_optimize
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=False,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=local_optimize,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-47.043)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-9.75,1.79]))<1e-2)
    
    def test_local_prior(self):
        "Test if the GP can be local optimized with prior distributions "
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_prior
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.pdistributions import Normal_prior
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=local_prior,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Construct the prior distributions of the hyperparameters
        pdis=dict(length=Normal_prior(mu=0.0,std=2.0),noise=Normal_prior(mu=-4.0,std=2.0))
        # Optimize the hyperparameters
        sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=pdis,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-47.042)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-16.38,1.79]))<1e-2)
    
    def test_local_ed_guess(self):
        "Test if the GP can be local optimized with educated guesses"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_ed_guess
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=local_ed_guess,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-47.042)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-15.39,1.79]))<1e-2)
    
    def test_random(self):
        "Test if the GP can be local optimized from random sampling"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import random
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(local_run=scipy_opt,maxiter=600,jac=True,npoints=10,local_kwargs=local_kwargs)
        # Define the list of arguments for the random sampling optimizer that are tested
        test_kwargs=[dict(bounds=None,hptrans=True,use_bounds=True),
                     dict(bounds=None,hptrans=True,use_bounds=False),
                     dict(bounds=None,hptrans=False,use_bounds=True),
                     dict(bounds=None,hptrans=False,use_bounds=False),
                     dict(bounds=np.array([[-3.0,3.0],[-8.0,0.0],[-2.0,4.0]]),hptrans=False,use_bounds=False)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':47.042,'x':np.array([1.97,-15.39,1.79])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])}]
        # Test the arguments for the random sampling optimizer
        for index,test_kwarg in enumerate(test_kwargs):
            with self.subTest(test_kwarg=test_kwarg):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=random,opt_kwargs=dict(**opt_kwargs,**test_kwarg))
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_grid(self):
        "Test if the GP can be optimized from grid search"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import grid
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,n_each_dim=5,local_kwargs=local_kwargs)
        # Define the list of arguments for the grid search optimizer that are tested
        test_kwargs=[dict(optimize=False,bounds=None,hptrans=True,use_bounds=True),
                     dict(optimize=True,bounds=None,hptrans=True,use_bounds=True),
                     dict(optimize=True,bounds=None,hptrans=True,use_bounds=False),
                     dict(optimize=True,bounds=None,hptrans=False,use_bounds=True),
                     dict(optimize=True,bounds=None,hptrans=False,use_bounds=False),
                     dict(optimize=True,bounds=np.array([[-3.0,3.0],[-8.0,0.0],[-2.0,4.0]]),hptrans=False,use_bounds=False)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':49.515,'x':np.array([1.75,-70.98,1.84])},
                  {'fun':47.042,'x':np.array([1.97,-70.98,1.79])},
                  {'fun':47.878,'x':np.array([3.87,-1.34,1.94])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':47.878,'x':np.array([3.87,-1.34,1.94])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])}]
        # Test the arguments for the grid search optimizer
        for index,test_kwarg in enumerate(test_kwargs):
            with self.subTest(test_kwarg=test_kwarg):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=grid,opt_kwargs=dict(**opt_kwargs,**test_kwarg))
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_line(self):
        "Test if the GP can be optimized from iteratively line search"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import line
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,n_each_dim=10,loops=3,local_kwargs=local_kwargs)
        # Define the list of arguments for the line search optimizer that are tested
        test_kwargs=[dict(optimize=False,bounds=None,hptrans=True,use_bounds=True),
                     dict(optimize=True,bounds=None,hptrans=True,use_bounds=True),
                     dict(optimize=True,bounds=None,hptrans=True,use_bounds=False),
                     dict(optimize=True,bounds=None,hptrans=False,use_bounds=True),
                     dict(optimize=True,bounds=None,hptrans=False,use_bounds=False),
                     dict(optimize=True,bounds=np.array([[-3.0,3.0],[-8.0,0.0],[-2.0,4.0]]),hptrans=False,use_bounds=False)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':47.442,'x':np.array([2.06,-70.98,1.98])},
                  {'fun':47.042,'x':np.array([1.97,-70.98,1.79])},
                  {'fun':47.878,'x':np.array([3.87,-1.34,1.94])},
                  {'fun':47.042,'x':np.array([1.97,-15.37,1.79])},
                  {'fun':47.878,'x':np.array([3.87,-1.34,1.94])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])}]
        # Test the arguments for the line search optimizer
        for index,test_kwarg in enumerate(test_kwargs):
            with self.subTest(test_kwarg=test_kwarg):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=line,opt_kwargs=dict(**opt_kwargs,**test_kwarg))
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_basin(self):
        "Test if the GP can be optimized from basin hopping"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.global_opt import basin
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(maxiter=500,jac=True,niter=5,interval=10,T=1.0,stepsize=0.1,niter_success=None,local_kwargs=local_kwargs)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=basin,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-47.042)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-15.58,1.79]))<1e-2)

    def test_annealling(self):
        "Test if the GP can be optimized from simulated annealling"
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.global_opt import annealling
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(maxiter=500,jac=False,initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False,local_kwargs=local_kwargs)
        # Define the list of arguments for the simulated annealling optimizer that are tested
        test_kwargs=[dict(bounds=None,use_bounds=True),
                     dict(bounds=None,use_bounds=False),
                     dict(bounds=np.array([[-3.0,3.0],[-8.0,0.0],[-2.0,4.0]]),use_bounds=True)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':47.042,'x':np.array([1.97,-14.39,1.79])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':47.648,'x':np.array([1.88,-7.37,1.81])}]
        # Test the arguments for the simulated annealling optimizer
        for index,test_kwarg in enumerate(test_kwargs):
            with self.subTest(test_kwarg=test_kwarg):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=annealling,opt_kwargs=dict(**opt_kwargs,**test_kwarg))
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_line_search_scale(self):
        "Test if the GP can be optimized from line search in the length-scale hyperparameter "
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.functions import calculate_list_values
        from catlearn.regression.gaussianprocess.optimizers.local_opt import run_golden,fine_grid_search,fine_grid_search_hptrans
        from catlearn.regression.gaussianprocess.optimizers.global_opt import line_search_scale
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.factorized_likelihood import FactorizedLogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(maxiter=500,jac=False,ngrid=80)
        # Define the list of arguments for the line search optimizer that are tested
        test_kwargs=[(dict(local_run=run_golden,bounds=None,hptrans=False,use_bounds=False),dict(fun_list=calculate_list_values,tol=1e-5,optimize=False,multiple_min=False)),
                     (dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=False),dict(fun_list=calculate_list_values,tol=1e-5,optimize=False,multiple_min=False)),
                     (dict(local_run=run_golden,bounds=None,hptrans=False,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,optimize=False,multiple_min=False)),
                     (dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,optimize=False,multiple_min=False)),
                     (dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,optimize=True,multiple_min=False)),
                     (dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,optimize=True,multiple_min=True)),
                     (dict(local_run=fine_grid_search,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,multiple_min=False)),
                     (dict(local_run=fine_grid_search,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,multiple_min=True)),
                     (dict(local_run=fine_grid_search,bounds=np.array([[-3.0,3.0],[-8.0,0.0],[-2.0,4.0]]),hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,multiple_min=True)),
                     (dict(local_run=fine_grid_search_hptrans,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,likelihood=False)),
                     (dict(local_run=fine_grid_search_hptrans,bounds=None,hptrans=True,use_bounds=True),dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,likelihood=True))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':44.713,'x':np.array([2.53,-1.98,1.67])},
                  {'fun':44.911,'x':np.array([2.46,-1.93,1.62])},
                  {'fun':44.703,'x':np.array([2.56,-2.00,1.70])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])}]
        # Test the arguments for the line search optimizer
        for index,(test_kwarg,local_kwargs) in enumerate(test_kwargs):
            with self.subTest(test_kwarg=test_kwarg,local_kwargs=local_kwargs):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimization_method=line_search_scale,opt_kwargs=dict(local_kwargs=local_kwargs,**test_kwarg,**opt_kwargs))
                # Construct the Gaussian process
                gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

if __name__ == '__main__':
    unittest.main()

