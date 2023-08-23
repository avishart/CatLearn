import numpy as np
from ..objectivefunction import ObjectiveFuction
from ...optimizers.local_opt import run_golden,fine_grid_search
from ...optimizers.functions import make_lines

class FactorizedLogLikelihood(ObjectiveFuction):
    def __init__(self,get_prior_mean=False,modification=False,ngrid=80,bounds=None,hptrans=True,use_bounds=True,s=0.14,noise_method="finegrid",method_kwargs={},**kwargs):
        """ The factorized log-likelihood objective function that is used to optimize the hyperparameters. 
            The prefactor hyperparameter is determined from an analytical expression. 
            An eigendecomposition is performed to get the eigenvalues. 
            The relative-noise hyperparameter can be searched from a single eigendecomposition for each length-scale hyperparameter. 
            Parameters:
                get_prior_mean: bool
                    Whether to save the parameters of the prior mean in the solution.
                modification: bool
                    Whether to modify the analytical prefactor value in the end.
                    The prefactor hyperparameter becomes larger if modification=True.
                ngrid: int
                    Number of grid points that are searched in the relative-noise hyperparameter. 
                bounds: Boundary_conditions class
                    A class that calculates the boundary conditions of the relative-noise hyperparameter.
                hptrans: bool
                    Whether to use a variable transformation of the relative-noise hyperparameter.
                use_bounds: bool
                    Whether to use educated guesses for the boundary conditions of the relative-noise hyperparameter.
                s: float
                    A value used by the variable transformation that determines how much extra space than the boundary conditions are used.
                noise_method: str or function
                    The method used for optimizing the relative-noise hyperparameter.
        """
        super().__init__(get_prior_mean=get_prior_mean,**kwargs)
        # Modification of the prefactor hyperparameter
        self.modification=modification
        # Parameters for construction of noise grid
        self.ngrid=ngrid
        self.bounds=bounds if bounds is None else bounds.copy()
        self.hptrans=hptrans
        self.use_bounds=use_bounds
        self.s=s
        # Method for finding the maximum noise hyperparameter
        self.noise_method=noise_method
        self.set_noise_optimizer(noise_method=noise_method,method_kwargs=method_kwargs)
    
    def set_noise_optimizer(self,noise_method="finegrid",method_kwargs={}):
        " Set method for finding the maximum noise hyperparameter. "
        self.method_kwargs=dict()
        if isinstance(noise_method,str):
            noise_method=noise_method.lower()
            if noise_method=="golden":
                self.maximize_noise=self.maximize_noise_golden
                self.method_kwargs=dict(tol=1e-5,maxiter=400,optimize=True,multiple_min=False)
            elif noise_method=="finegrid":
                self.maximize_noise=self.maximize_noise_finegrid
                self.method_kwargs=dict(tol=1e-5,maxiter=400,loops=2,iterloop=80,optimize=True,multiple_min=False)
            elif noise_method=="grid":
                self.maximize_noise=self.maximize_noise_grid
        else:
            self.maximize_noise=noise_method
        self.method_kwargs.update(method_kwargs)
        return self

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(model,X,Y)
        noise,nlp=self.maximize_noise(model,X,Y_p,hp.copy(),parameters_set,parameters,pdis,UTY,D,n_data)
        if jac:
            deriv=self.derivative(hp,parameters_set,parameters,model,X,KXX,D,U,Y_p,UTY,noise,pdis,**kwargs)
            self.update_solution(nlp,theta,parameters,model,jac=jac,deriv=deriv,noise=noise,UTY=UTY,D=D,n_data=n_data)
            return nlp,deriv
        self.update_solution(nlp,theta,parameters,model,jac=jac,noise=noise,UTY=UTY,D=D,n_data=n_data)
        return nlp
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,D,U,Y_p,UTY,noise,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        nlp_deriv=np.array([])
        D_n=D+np.exp(2.0*noise)
        prefactor2=np.mean(UTY/D_n)
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        KXX_inv=np.matmul(U/D_n,U.T)
        coef=np.matmul(KXX_inv,Y_p)
        for para in parameters_set:
            if para=='prefactor':
                nlp_deriv=np.append(nlp_deriv,np.zeros((len(hp[para]))))
                continue
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv)
            nlp_deriv=np.append(nlp_deriv,(-(0.5*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1))/prefactor2)+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)
        return nlp_deriv
    
    def get_eig_ll(self,noise,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Calculate log-likelihood from Eigendecomposition for a noise value. "
        D_n=D+np.exp(2*noise)
        prefactor2=np.mean(UTY/D_n)
        nlp=0.5*n_data*(np.log(prefactor2)+1+np.log(2*np.pi))+0.5*np.sum(np.log(D_n))
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        return nlp-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
    
    def get_all_eig_ll(self,noises,fun,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Calculate log-likelihood from Eigendecompositions for all noise values from the list. "
        D_n=D+np.exp(2*noises)
        prefactor=0.5*np.log(np.mean(UTY/D_n,axis=1))
        nlp=(0.5*n_data*(1+np.log(2.0*np.pi)))+((n_data*prefactor)+(0.5*np.sum(np.log(D_n),axis=1)))
        hp.update(dict(prefactor=prefactor.reshape(-1,1),noise=noises))
        return nlp-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
    
    def maximize_noise_golden(self,model,X,Y,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method combined with the golden section search method for local optimization. "
        noises=self.make_noise_list(model,X,Y)
        args_ll=(hp,parameters_set,parameters,pdis,UTY,D,n_data)
        sol=run_golden(self.get_eig_ll,noises,fun_list=self.get_all_eig_ll,args=args_ll,**self.method_kwargs)
        return sol['x'],sol['fun']
    
    def maximize_noise_finegrid(self,model,X,Y,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method combined with a finer grid method for local optimization. "
        noises=self.make_noise_list(model,X,Y)
        args_ll=(hp,parameters_set,parameters,pdis,UTY,D,n_data)
        sol=fine_grid_search(self.get_eig_ll,noises,fun_list=self.get_all_eig_ll,args=args_ll,**self.method_kwargs)
        return sol['x'],sol['fun']
    
    def maximize_noise_grid(self,model,X,Y,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method. "
        noises=self.make_noise_list(model,X,Y)
        # Calculate function values for line coordinates
        f_list=self.get_all_eig_ll(noises,self.get_eig_ll,hp,parameters_set,parameters,pdis,UTY,D,n_data)
        # Find the minimum value
        i_min=np.nanargmin(f_list)
        return noises[i_min],f_list[i_min]
    
    def make_noise_list(self,model,X,Y,**kwargs):
        " Make the list of noises. " 
        return np.array(make_lines(['noise'],model,X,Y,bounds=self.bounds,ngrid=self.ngrid,hptrans=self.hptrans,use_bounds=self.use_bounds,ngrid_each_dim=False,s=self.s)).reshape(-1,1)
    
    def update_solution(self,fun,theta,parameters,model,jac=False,deriv=None,noise=None,UTY=None,D=None,n_data=None,**kwargs):
        " Update the solution of the optimization in terms of hyperparameters and model. "
        if fun<self.sol['fun']:
            hp,parameters_set=self.make_hp(theta,parameters)
            D_n=D+np.exp(2.0*noise)
            prefactor2=np.mean(UTY/D_n)
            if self.modification:
                prefactor2=(n_data/(n_data-len(theta)))*prefactor2 if n_data-len(theta)>0 else prefactor2
            hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]),noise=np.array([noise]).reshape(-1)))
            self.sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol
    
    def copy(self):
        " Copy the objective function object. "
        clone=self.__class__(get_prior_mean=self.get_prior_mean,modification=self.modification,
                             ngrid=self.ngrid,bounds=self.bounds,
                             hptrans=self.hptrans,use_bounds=self.use_bounds,
                             s=self.s,noise_method=self.noise_method,method_kwargs=self.method_kwargs)
        return clone
    
    def __repr__(self):
        return "{}(get_prior_mean={},modification={},ngrid={},hptrans={},use_bounds={},s={},noise_method={finegrid})".format(self.__class__.__name__,self.get_prior_mean,self.modification,self.ngrid,self.hptran,self.use_bounds,self.s,self.noise_method)
