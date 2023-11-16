import numpy as np
from scipy.linalg import cho_solve
from .loo import LOO

class GPP(LOO):
    def __init__(self,get_prior_mean=False,**kwargs):
        """ 
        The Geissers surrogate predictive probability objective function as a function of the hyperparameters.
        The prefactor hyperparameter is calculated from an analytical expression. 
        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean in the solution.
        """
        # Set descriptor of the objective function
        self.use_analytic_prefactor=True
        self.use_optimized_noise=False
        # Set the arguments
        self.update_arguments(get_prior_mean=get_prior_mean,**kwargs)

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update_model(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        KXX_inv,K_inv_diag,coef_re,co_Kinv=self.get_co_Kinv(L,low,n_data,coef)
        prefactor2=np.mean(co_Kinv*coef_re)
        prefactor=0.5*np.log(prefactor2)
        hp['prefactor']=np.array([prefactor])
        gpp_v=1.0-np.mean(np.log(K_inv_diag))+2.0*prefactor+np.log(2.0*np.pi)
        gpp_v=gpp_v-self.logpriors(hp,pdis,jac=False)/n_data
        if jac:
            deriv=self.derivative(hp,parameters_set,model,X,KXX,KXX_inv,K_inv_diag,coef_re,co_Kinv,prefactor2,n_data,pdis,**kwargs)  
            self.update_solution(gpp_v,theta,hp,model,jac=jac,deriv=deriv,prefactor2=prefactor2)
            return gpp_v,deriv
        self.update_solution(gpp_v,theta,hp,model,jac=jac,prefactor2=prefactor2)
        return gpp_v
    
    def derivative(self,hp,parameters_set,model,X,KXX,KXX_inv,K_inv_diag,coef_re,co_Kinv,prefactor2,n_data,pdis,**kwargs):
        gpp_deriv=np.array([])
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1)))  
        for para in parameters_set:
            if para=='prefactor':
                gpp_deriv=np.append(gpp_deriv,np.zeros((len(hp[para]))))
                continue
            K_deriv=self.get_K_deriv(model,para,X=X,KXX=KXX)
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef_re)
            gpp_d=(np.mean(co_Kinv*(2.0*r_j+co_Kinv*s_j),axis=-1)/prefactor2)+np.mean(s_j/K_inv_diag,axis=-1)
            gpp_deriv=np.append(gpp_deriv,gpp_d)
        gpp_deriv=gpp_deriv-self.logpriors(hp,pdis,jac=True)/n_data
        return gpp_deriv
    
    def update_arguments(self,get_prior_mean=None,**kwargs):
        """
        Update the objective function with its arguments. The existing arguments are used if they are not given.
        Parameters:
            get_prior_mean : bool
                Whether to get the parameters of the prior mean in the solution.
        Returns:
            self: The updated object itself.
        """
        if get_prior_mean is not None:
            self.get_prior_mean=get_prior_mean
        # Always reset the solution when the objective function is changed 
        self.reset_solution()
        return self
    
    def update_solution(self,fun,theta,hp,model,jac=False,deriv=None,prefactor2=None,**kwargs):
        """
        Update the solution of the optimization in terms of hyperparameters and model.
        The lowest objective function value is stored togeher with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        The prefactor hyperparameter are stored as a different value
        than the input since it is optimized analytically.
        """
        if fun<self.sol['fun']:
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
        arg_kwargs=dict(get_prior_mean=self.get_prior_mean)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    