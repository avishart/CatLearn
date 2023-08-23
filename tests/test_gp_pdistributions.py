import unittest
import numpy as np
from .functions import create_func,make_train_test_set

class TestGPPdis(unittest.TestCase):
    """ Test if the Gaussian Process can be optimized with all existing prior distributions of the hyperparameters. """
    
    def test_local_prior(self):
        "Test if the GP can be local optimized with prior distributions "
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.local_opt import scipy_opt
        from catlearn.regression.gaussianprocess.optimizers.global_opt import local_prior
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.likelihood import LogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.pdistributions import Uniform_prior,Normal_prior,Gen_normal_prior,Gamma_prior,Invgamma_prior
        from catlearn.regression.gaussianprocess.pdistributions.pdistributions import make_prior
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        opt_kwargs=dict(local_run=scipy_opt,maxiter=500,jac=True,local_kwargs=local_kwargs)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=LogLikelihood(),optimization_method=local_prior,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Define the list of prior distribution objects that are tested
        test_pdis=[(False,Uniform_prior(start=[-9.0],end=[9.0],prob=[1.0/18.0])),
                   (True,Uniform_prior(start=[-9.0],end=[9.0],prob=[1.0/18.0])),
                   (False,Normal_prior(mu=[0.0],std=[3.0])),
                   (True,Normal_prior(mu=[0.0],std=[3.0])),
                   (False,Gen_normal_prior(mu=[0.0],s=[3.0],v=[2])),
                   (True,Gen_normal_prior(mu=[0.0],s=[3.0],v=[2])),
                   (False,Gamma_prior(a=[1e-5],b=[1e-5])),
                   (True,Gamma_prior(a=[1e-5],b=[1e-5])),
                   (False,Invgamma_prior(a=[1e-5],b=[1e-5])),
                   (True,Invgamma_prior(a=[1e-5],b=[1e-5]))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':47.042,'x':np.array([1.97,-15.39,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-15.27,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-16.20,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-16.18,1.79])},
                  {'fun':44.702,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':47.042,'x':np.array([1.97,-17.15,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-11.43,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-17.17,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-17.32,1.79])},
                  {'fun':47.042,'x':np.array([1.97,-17.24,1.79])}]
        # Test the prior distributions
        for index,(make_pdis,pdis_d) in enumerate(test_pdis):
            with self.subTest(make_pdis=make_pdis,pdis_d=pdis_d):
                # Construct the prior distribution objects
                pdis=dict(length=pdis_d.copy(),noise=pdis_d.copy())
                # Use educated guesses for the prior distributions if make_pdis==True
                if make_pdis:
                    pdis=make_prior(gp,['length','noise'],x_tr,f_tr,prior_dis=pdis,scale=1)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=pdis,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)

    def test_global_prior(self):
        "Test if the GP can be global optimized with prior distributions "
        from catlearn.regression.gaussianprocess.models.gp import GaussianProcess
        from catlearn.regression.gaussianprocess.optimizers.functions import calculate_list_values
        from catlearn.regression.gaussianprocess.optimizers.local_opt import fine_grid_search
        from catlearn.regression.gaussianprocess.optimizers.global_opt import line_search_scale
        from catlearn.regression.gaussianprocess.objectivefunctions.gp.factorized_likelihood import FactorizedLogLikelihood
        from catlearn.regression.gaussianprocess.hpfitter import HyperparameterFitter
        from catlearn.regression.gaussianprocess.pdistributions import Uniform_prior,Normal_prior,Gen_normal_prior,Gamma_prior,Invgamma_prior
        from catlearn.regression.gaussianprocess.pdistributions.pdistributions import make_prior
        # Create the data set
        x,f,g=create_func()
        ## Whether to learn from the derivatives
        use_derivatives=False
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Make the dictionary of the optimization
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,loops=3,iterloop=80,optimize=True,multiple_min=True)
        opt_kwargs=dict(local_run=fine_grid_search,bounds=None,hptrans=True,use_bounds=True,maxiter=500,jac=False,ngrid=80,local_kwargs=local_kwargs)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimization_method=line_search_scale,opt_kwargs=opt_kwargs)
        # Construct the Gaussian process
        gp=GaussianProcess(hp=dict(length=2.0),hpfitter=hpfitter,use_derivatives=use_derivatives)
        # Define the list of prior distribution objects that are tested
        test_pdis=[(False,Uniform_prior(start=[-9.0],end=[9.0],prob=[1.0/18.0])),
                   (True,Uniform_prior(start=[-9.0],end=[9.0],prob=[1.0/18.0])),
                   (False,Normal_prior(mu=[0.0],std=[3.0])),
                   (True,Normal_prior(mu=[0.0],std=[3.0])),
                   (False,Gen_normal_prior(mu=[0.0],s=[3.0],v=[2])),
                   (True,Gen_normal_prior(mu=[0.0],s=[3.0],v=[2])),
                   (False,Gamma_prior(a=[1e-5],b=[1e-5])),
                   (True,Gamma_prior(a=[1e-5],b=[1e-5])),
                   (False,Invgamma_prior(a=[1e-5],b=[1e-5])),
                   (True,Invgamma_prior(a=[1e-5],b=[1e-5]))]
        # Make a list of the solution values that the test compares to
        sol_list=[{'fun':1464.267,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':49.284,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':49.315,'x':np.array([2.54,-1.96,1.67])},
                  {'fun':50.481,'x':np.array([2.55,-2.00,1.69])},
                  {'fun':48.905,'x':np.array([2.53,-1.94,1.65])},
                  {'fun':51.039,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':67.728,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':49.743,'x':np.array([2.55,-1.93,1.67])},
                  {'fun':67.728,'x':np.array([2.55,-1.99,1.69])},
                  {'fun':79.960,'x':np.array([2.52,-1.97,1.66])}]
        # Test the prior distributions
        for index,(make_pdis,pdis_d) in enumerate(test_pdis):
            with self.subTest(make_pdis=make_pdis,pdis_d=pdis_d):
                # Construct the prior distribution objects
                pdis=dict(length=pdis_d.copy(),noise=pdis_d.copy())
                # Use educated guesses for the prior distributions if make_pdis==True
                if make_pdis:
                    pdis=make_prior(gp,['length','noise'],x_tr,f_tr,prior_dis=pdis,scale=1)
                # Optimize the hyperparameters
                sol=gp.optimize(x_tr,f_tr,retrain=False,hp=None,pdis=pdis,verbose=False)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-sol_list[index]['fun'])<1e-2) 
                self.assertTrue(np.linalg.norm(sol['x']-sol_list[index]['x'])<1e-2)


if __name__ == '__main__':
    unittest.main()

