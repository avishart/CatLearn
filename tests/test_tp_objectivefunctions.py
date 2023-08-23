import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestTPObjectiveFunctions(unittest.TestCase):
    """ Test if the Student T Process can be optimized with all existing objective functions. """
    
    def test_local(self):
        "Test if the TP can be local optimized with all objective functions that does not use eigendecomposition"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_optimize
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.objectivefunctions.tp import LogLikelihood
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=dict(tol=1e-12,method='L-BFGS-B'))
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=local_optimize,opt_kwargs=opt_kwargs)
        # Construct the Student t process
        tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Optimize the hyperparameters
        sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-502.207)<1e-2) 
        self.assertTrue(np.linalg.norm(sol['x']-np.array([1.97,-14.64]))<1e-2)
    
    def test_line_search_scale(self):
        "Test if the TP can be optimized from line search in the length-scale hyperparameter with all objective functions that uses eigendecomposition"
        from catlearn.regression.gaussianprocess.models.tp import TProcess
        from catlearn.regression.gaussianprocess.optimizers.functions import calculate_list_values
        from catlearn.regression.gaussianprocess.optimizers.local_opt import fine_grid_search
        from catlearn.regression.gaussianprocess.optimizers.global_opt import line_search_scale
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.objectivefunctions.tp import FactorizedLogLikelihood,FactorizedLogLikelihoodSVD
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Define the list of objective function objects that are tested
        obj_list=[FactorizedLogLikelihood(modification=False,ngrid=250,bounds=None,hptrans=True,use_bounds=True,s=0.14,noise_method="grid",method_kwargs={}),
                  FactorizedLogLikelihood(modification=True,ngrid=250,bounds=None,hptrans=True,use_bounds=True,s=0.14,noise_method="grid",method_kwargs={}),
                  FactorizedLogLikelihood(modification=False,ngrid=80,bounds=None,hptrans=True,use_bounds=True,s=0.14,noise_method="golden",method_kwargs={}),
                  FactorizedLogLikelihood(modification=False,ngrid=80,bounds=None,hptrans=True,use_bounds=True,s=0.14,noise_method="finegrid",method_kwargs={}),
                  FactorizedLogLikelihood(modification=True,ngrid=250,bounds=None,hptrans=True,use_bounds=False,s=0.14,noise_method="grid",method_kwargs={}),
                  FactorizedLogLikelihood(modification=True,ngrid=250,bounds=None,hptrans=False,use_bounds=True,s=0.14,noise_method="grid",method_kwargs={}),
                  FactorizedLogLikelihood(modification=True,ngrid=250,bounds=np.array([[-3.0,3.0],[-8.0,0.0],[-2.0,4.0]]),hptrans=True,use_bounds=True,s=0.14,noise_method="grid",method_kwargs={}),
                  FactorizedLogLikelihoodSVD(modification=False,ngrid=250,bounds=None,hptrans=True,use_bounds=True,s=0.14,noise_method="grid",method_kwargs={})]
        # Make the dictionary of the optimization
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,multiple_min=True)
        opt_kwargs=dict(local_run=fine_grid_search,bounds=None,hptrans=True,use_bounds=True,maxiter=500,jac=False,ngrid=80,local_kwargs=local_kwargs)
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':499.867,'x':np.array([2.55,-2.01])},
                  {'fun':499.867,'x':np.array([2.55,-2.01])},
                  {'fun':499.866,'x':np.array([2.55,-2.00])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.869,'x':np.array([2.55,-2.03])},
                  {'fun':499.868,'x':np.array([2.55,-2.02])},
                  {'fun':499.866,'x':np.array([2.55,-1.99])},
                  {'fun':499.867,'x':np.array([2.55,-2.01])}]
        # Test the objective function objects
        for index,obj_func in enumerate(obj_list):
            with self.subTest(obj_func=obj_func):
                # Construct the hyperparameter fitter
                hpfitter=HyperparameterFitter(func=obj_func,optimization_method=line_search_scale,opt_kwargs=opt_kwargs)
                # Construct the Student t process
                tp=TProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
                # Optimize the hyperparameters
                sol=tp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=None,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)


if __name__ == '__main__':
    unittest.main()

