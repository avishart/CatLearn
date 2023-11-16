import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestTPOptimizer(unittest.TestCase):
    """ Test if the Student t Process can be optimized with all existing optimization methods and objective functions. """

    def test_function(self):
        "Test if the function value of the LogLikelihood is obtainable "
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.optimizer import FunctionEvaluation
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=FunctionEvaluation(jac=True))
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.256)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([2.00,-8.00]))<1e-2)
    
    def test_local_jac(self):
        "Test if the TP can be local optimized with gradients wrt the hyperparameters"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the optimizer
        optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.207)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-14.64]))<1e-2)

    def test_local_nojac(self):
        "Test if the TP can be local optimized without gradients wrt the hyperparameters"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the optimizer
        optimizer=ScipyOptimizer(maxiter=500,jac=False,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.207)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-9.60]))<1e-2)
    
    def test_local_prior(self):
        "Test if the TP can be local optimized with prior distributions "
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyPriorOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.pdistributions import Normal_prior
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the optimizer
        optimizer=ScipyPriorOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Construct the prior distributions of the hyperparameters
        pdis=dict(length=Normal_prior(mu=0.0,std=2.0),noise=Normal_prior(mu=-4.0,std=2.0))
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=pdis,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.207)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-13.85]))<1e-2)
    
    def test_local_ed_guess(self):
        "Test if the TP can be local optimized with educated guesses"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyGuessOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.strict import StrictBoundaries
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the optimizer
        optimizer=ScipyGuessOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=StrictBoundaries())
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.207)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-16.09]))<1e-2)
    
    def test_random(self):
        "Test if the TP can be local optimized from random sampling"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import RandomSamplingOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.educated import EducatedBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the local optimizer
        local_optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make the global optimizer
        optimizer=RandomSamplingOptimizer(local_optimizer=local_optimizer,maxiter=600,npoints=10,parallel=False)
        # Make fixed boundary conditions for one of the tests
        bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]])
        # Define the list of arguments for the random sampling optimizer that are tested
        bounds_list=[VariableTransformation(bounds=None),
                     EducatedBoundaries(),
                     HPBoundaries(bounds_dict=bounds_dict)]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])}]
        # Test the arguments for the random sampling optimizer
        for index,bounds in enumerate(bounds_list):
            with self.subTest(bounds=bounds):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_grid(self):
        "Test if the TP can be optimized from grid search"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import GridOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.educated import EducatedBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the local optimizer
        local_optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make the global optimizer kwargs
        opt_kwargs=dict(maxiter=500,n_each_dim=5,parallel=False)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        bounds_ed=EducatedBoundaries()
        fixed_bounds=HPBoundaries(bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]]),log=True)
        # Define the list of arguments for the grid search optimizer that are tested
        test_kwargs=[(bounds_trans,GridOptimizer(local_optimizer=local_optimizer,optimize=False,**opt_kwargs)),
                     (bounds_trans,GridOptimizer(local_optimizer=local_optimizer,optimize=True,**opt_kwargs)),
                     (bounds_ed,GridOptimizer(local_optimizer=local_optimizer,optimize=True,**opt_kwargs)),
                     (fixed_bounds,GridOptimizer(local_optimizer=local_optimizer,optimize=True,**opt_kwargs))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':502.256,'x':np.array([2.00,-8.00])},
                  {'fun':502.207,'x':np.array([1.97,-14.64])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':502.207,'x':np.array([1.97,-14.64])}]
        # Test the arguments for the grid search optimizer
        for index,(bounds,optimizer) in enumerate(test_kwargs):
            with self.subTest(bounds=bounds,optimizer=optimizer):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_line(self):
        "Test if the TP can be optimized from iteratively line search"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.localoptimizer import ScipyOptimizer
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import IterativeLineOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.educated import EducatedBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the local optimizer
        local_optimizer=ScipyOptimizer(maxiter=500,jac=True,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Make the global optimizer kwargs
        opt_kwargs=dict(maxiter=500,n_each_dim=10,loops=3,parallel=False)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        bounds_ed=EducatedBoundaries()
        fixed_bounds=HPBoundaries(bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]]),log=True)
        # Define the list of arguments for the line search optimizer that are tested
        test_kwargs=[(bounds_trans,IterativeLineOptimizer(local_optimizer=local_optimizer,optimize=False,**opt_kwargs)),
                     (bounds_trans,IterativeLineOptimizer(local_optimizer=local_optimizer,optimize=True,**opt_kwargs)),
                     (bounds_ed,IterativeLineOptimizer(local_optimizer=local_optimizer,optimize=True,**opt_kwargs)),
                     (fixed_bounds,IterativeLineOptimizer(local_optimizer=local_optimizer,optimize=True,**opt_kwargs))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':502.561,'x':np.array([2.06,-70.98])},
                  {'fun':502.207,'x':np.array([1.97,-70.98])},
                  {'fun':502.207,'x':np.array([1.97,-15.37])},
                  {'fun':502.207,'x':np.array([1.97,-16.21])}]
        # Test the arguments for the line search optimizer
        for index,(bounds,optimizer) in enumerate(test_kwargs):
            with self.subTest(bounds=bounds,optimizer=optimizer):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_basin(self):
        "Test if the TP can be optimized from basin hopping"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import BasinOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(niter=5,interval=10,T=1.0,stepsize=0.1,niter_success=None)
        # Make the optimizer
        optimizer=BasinOptimizer(maxiter=500,jac=True,opt_kwargs=opt_kwargs,local_kwargs=local_kwargs)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Set random seed to give the same results every time
        np.random.seed(1)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.207)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-14.71]))<1e-2)

    def test_annealling(self):
        "Test if the TP can be optimized from simulated annealling"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import AnneallingOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.educated import EducatedBoundaries
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False)
        # Make the optimizer
        optimizer=AnneallingOptimizer(maxiter=500,jac=False,opt_kwargs=opt_kwargs,local_kwargs=local_kwargs)
        # Make the boundary conditions for the tests
        bounds_ed=EducatedBoundaries()
        fixed_bounds=HPBoundaries(bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]]),log=True)
        # Define the list of arguments for the simulated annealling optimizer that are tested
        bounds_list=[bounds_ed,fixed_bounds]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])}]
        # Test the arguments for the simulated annealling optimizer
        for index,bounds in enumerate(bounds_list):
            with self.subTest(bounds=bounds):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_annealling_trans(self):
        "Test if the TP can be optimized from simulated annealling with variable transformation. "
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import AnneallingTransOptimizer
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(initial_temp=5230.0,restart_temp_ratio=2e-05,visit=2.62,accept=-5.0,seed=None,no_local_search=False)
        # Make the optimizer
        optimizer=AnneallingTransOptimizer(maxiter=500,jac=False,opt_kwargs=opt_kwargs,local_kwargs=local_kwargs)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        fixed_bounds=HPBoundaries(bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]]),log=True)
        bounds_fixed_trans=VariableTransformation(bounds=fixed_bounds)
        # Define the list of arguments for the simulated annealling optimizer that are tested
        bounds_list=[bounds_trans,bounds_fixed_trans]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])}]
        # Test the arguments for the simulated annealling optimizer
        for index,bounds in enumerate(bounds_list):
            with self.subTest(bounds=bounds):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=LogLikelihood(),optimizer=optimizer,bounds=bounds)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_line_search_scale(self):
        "Test if the TP can be optimized from line search in the length-scale hyperparameter "
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.globaloptimizer import FactorizedOptimizer
        from catlearn.regression.gaussianprocess.optimizers.linesearcher import GoldenSearch,FineGridSearch,TransGridSearch
        from catlearn.regression.gaussianprocess.objectivefunctions.tp.factorized_likelihood import FactorizedLogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.hpboundary.boundary import HPBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.educated import EducatedBoundaries
        from catlearn.regression.gaussianprocess.hpboundary.hptrans import VariableTransformation
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(maxiter=500,jac=False,tol=1e-5,parallel=False)
        # Make the boundary conditions for the tests
        bounds_trans=VariableTransformation(bounds=None)
        bounds_ed=EducatedBoundaries()
        fixed_bounds=HPBoundaries(bounds_dict=dict(length=[[-3.0,3.0]],noise=[[-8.0,0.0]],prefactor=[[-2.0,4.0]]),log=True)
        # Define the list of arguments for the line search optimizer that are tested
        test_kwargs=[(bounds_ed,GoldenSearch(optimize=False,multiple_min=False,**opt_kwargs)),
                     (bounds_trans,GoldenSearch(optimize=False,multiple_min=False,**opt_kwargs)),
                     (bounds_trans,GoldenSearch(optimize=True,multiple_min=False,**opt_kwargs)),
                     (bounds_trans,GoldenSearch(optimize=True,multiple_min=True,**opt_kwargs)),
                     (bounds_trans,FineGridSearch(optimize=True,multiple_min=False,loops=3,ngrid=80,**opt_kwargs)),
                     (bounds_trans,FineGridSearch(optimize=True,multiple_min=True,loops=3,ngrid=80,**opt_kwargs)),
                     (fixed_bounds,FineGridSearch(optimize=True,multiple_min=True,loops=3,ngrid=80,**opt_kwargs)),
                     (bounds_trans,TransGridSearch(optimize=True,use_likelihood=False,loops=3,ngrid=80,**opt_kwargs)),
                     (bounds_trans,TransGridSearch(optimize=True,use_likelihood=True,loops=3,ngrid=80,**opt_kwargs))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':499.867,'x':np.array([2.56,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])}]
        # Test the arguments for the line search optimizer
        for index,(bounds,line_optimizer) in enumerate(test_kwargs):
            with self.subTest(bounds=bounds,line_optimizer=line_optimizer):
                # Make the optimizer
                optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,bounds=bounds,ngrid=80)
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimizer=optimizer)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Set random seed to give the same results every time
                np.random.seed(1)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

if __name__ == '__main__':
    unittest.main()

