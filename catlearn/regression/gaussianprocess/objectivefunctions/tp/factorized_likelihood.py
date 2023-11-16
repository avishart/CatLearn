import numpy as np
from ..gp.factorized_likelihood import FactorizedLogLikelihood

class FactorizedLogLikelihood(FactorizedLogLikelihood):
    def __init__(self,get_prior_mean=False,ngrid=80,bounds=None,noise_optimizer=None,**kwargs):
        """ 
        The factorized log-likelihood objective function that is used to optimize the hyperparameters. 
        An eigendecomposition is performed to get the eigenvalues. 
        The relative-noise hyperparameter can be searched from a single eigendecomposition for each length-scale hyperparameter. 
        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean in the solution.
            ngrid: int
                Number of grid points that are searched in the relative-noise hyperparameter. 
            bounds: Boundary_conditions class
                A class of the boundary conditions of the relative-noise hyperparameter.
            noise_optimizer : Noise line search optimizer class
                A line search optimization method for the relative-noise hyperparameter.
        """
        # Set descriptor of the objective function
        self.use_analytic_prefactor=False
        self.use_optimized_noise=True
        # Set default bounds
        if bounds is None:
            from ...hpboundary.hptrans import VariableTransformation
            bounds=VariableTransformation(bounds=None)
        # Set default noise line optimizer
        if noise_optimizer is None:
            from ...optimizers.noisesearcher import NoiseFineGridSearch
            noise_optimizer=NoiseFineGridSearch(maxiter=1000,tol=1e-5,optimize=True,multiple_min=False,ngrid=ngrid,loops=2)
        # Set the arguments
        self.update_arguments(get_prior_mean=get_prior_mean,
                              ngrid=ngrid,
                              bounds=bounds,
                              noise_optimizer=noise_optimizer,
                              **kwargs)

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update_model(model,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(model,X,Y)
        noise,nlp=self.maximize_noise(parameters,model,X,Y,pdis,hp,UTY,D,n_data)
        if jac:
            deriv=self.derivative(hp,parameters_set,model,X,KXX,D,U,Y_p,UTY,noise,pdis,n_data,**kwargs)
            self.update_solution(nlp,theta,hp,model,jac=jac,deriv=deriv,noise=noise)
            return nlp,deriv
        self.update_solution(nlp,theta,hp,model,jac=jac,noise=noise)
        return nlp
    
    def derivative(self,hp,parameters_set,model,X,KXX,D,U,Y_p,UTY,noise,pdis,n_data,**kwargs):
        nlp_deriv=np.array([])
        D_n=D+np.exp(2*noise)
        hp['noise']=np.array([noise])
        KXX_inv=np.matmul(U/D_n,U.T)
        coef=np.matmul(KXX_inv,Y_p)
        a,b=self.get_hyperprior_parameters(model)
        ycoef=1.0+np.sum(UTY/D_n)/(2.0*b)
        for para in parameters_set:
            K_deriv=self.get_K_deriv(model,para,X=X,KXX=KXX)
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv)
            nlp_deriv=np.append(nlp_deriv,-0.5*((a+0.5*n_data)/b)*(np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)/ycoef)+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,pdis,jac=True)
        return nlp_deriv
    
    def get_eig_fun(self,noise,hp,pdis,UTY,D,n_data,a,b,**kwargs):
        " Calculate log-likelihood from Eigendecomposition for a noise value. "
        D_n=D+np.exp(2.0*noise)
        nlp=0.5*np.sum(np.log(D_n))+0.5*(2.0*a+n_data)*np.log(1.0+np.sum(UTY/D_n)/(2.0*b))
        if pdis is not None:
            hp['noise']=np.array([noise]).reshape(-1)
        return nlp-self.logpriors(hp,pdis,jac=False)
    
    def get_all_eig_fun(self,noises,hp,pdis,UTY,D,n_data,a,b,**kwargs):
        " Calculate log-likelihood from Eigendecompositions for all noise values from the list. "
        D_n=D+np.exp(2.0*noises)
        nlp=0.5*np.sum(np.log(D_n),axis=1)+0.5*(2.0*a+n_data)*np.log(1.0+np.sum(UTY/D_n,axis=1)/(2.0*b))
        if pdis is not None:
            hp['noise']=noises
        return nlp-self.logpriors(hp,pdis,jac=False)
    
    def maximize_noise(self,parameters,model,X,Y,pdis,hp,UTY,D,n_data,**kwargs):
        " Find the maximum relative-noise with a grid method. "
        noises=self.make_noise_list(model,X,Y)
        # Get the hyperprior parameters
        a,b=self.get_hyperprior_parameters(model)
        # Make the function arguments
        func_args=(hp.copy(),pdis,UTY,D,n_data,a,b)
        # Calculate function values for line coordinates
        sol=self.noise_optimizer.run(self,noises,['noise'],model,X,Y,pdis,func_args=func_args)
        # Find the minimum value
        return sol['x'][0],sol['fun']
    
    def get_hyperprior_parameters(self,model,**kwargs):
        " Get the hyperprior parameters from the Student's T Process. "
        a,b=model.get_hyperprior_parameters()
        return a,b

    def update_solution(self,fun,theta,hp,model,jac=False,deriv=None,noise=None,**kwargs):
        """
        Update the solution of the optimization in terms of hyperparameters and model.
        The lowest objective function value is stored togeher with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        The relative-noise hyperparameter is stored as different values
        than the input since they are optimized numerically.
        """
        if fun<self.sol['fun']:
            hp['noise']=np.array([noise])  
            self.sol['x']=np.concatenate([hp[para] for para in sorted(hp.keys())])
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=self.get_prior_parameters(model)
        return self.sol
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(get_prior_mean=self.get_prior_mean,
                        ngrid=self.ngrid,
                        bounds=self.bounds,
                        noise_optimizer=self.noise_optimizer)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
