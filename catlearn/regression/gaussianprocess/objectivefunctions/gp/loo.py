import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction

class LOO(ObjectiveFuction):
    def __init__(self,get_prior_mean=False,modification=False,**kwargs):
        """ The leave-one-out objective function that is used to optimize the hyperparameters. 
            Parameters:
                get_prior_mean: bool
                    Whether to save the parameters of the prior mean in the solution.
                modification: bool
                    Whether to calculate the analytical prefactor value in the end.
        """
        super().__init__(get_prior_mean=get_prior_mean,**kwargs)
        self.modification=modification
    
    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        KXX_inv,K_inv_diag,coef_re,co_Kinv=self.get_co_Kinv(L,low,n_data,coef)
        loo_v=np.mean(co_Kinv**2)
        loo_v=loo_v-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)/n_data
        if jac:
            deriv=self.derivative(hp,parameters_set,parameters,model,X,KXX,KXX_inv,K_inv_diag,coef_re,co_Kinv,n_data,pdis,**kwargs)
            self.update_solution(loo_v,theta,parameters,model,jac=jac,deriv=deriv,coef_re=coef_re,K_inv_diag=K_inv_diag,co_Kinv=co_Kinv)
            return loo_v,deriv
        self.update_solution(loo_v,theta,parameters,model,jac=jac,coef_re=coef_re,K_inv_diag=K_inv_diag,co_Kinv=co_Kinv)
        return loo_v
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,KXX_inv,K_inv_diag,coef_re,co_Kinv,n_data,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        loo_deriv=np.array([])
        for para in parameters_set:
            if para=='prefactor':
                loo_deriv=np.append(loo_deriv,np.zeros((len(hp[para]))))
                continue
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef_re)
            loo_deriv=np.append(loo_deriv,2.0*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv),axis=-1))
        loo_deriv=loo_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)/n_data
        return loo_deriv
    
    def get_co_Kinv(self,L,low,n_data,coef):
        " Get the analytic solution to the prefactor "
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        coef_re=coef.reshape(-1)
        co_Kinv=coef_re/K_inv_diag
        return KXX_inv,K_inv_diag,coef_re,co_Kinv
    
    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        r_j=np.einsum('ji,di->dj',KXX_inv,np.matmul(K_deriv,-coef))
        s_j=np.einsum('ji,dji->di',KXX_inv,np.matmul(K_deriv,KXX_inv))
        return r_j,s_j
    
    def update_solution(self,fun,theta,parameters,model,jac=False,deriv=None,coef_re=None,K_inv_diag=None,co_Kinv=None,**kwargs):
        " Update the solution of the optimization in terms of hyperparameters and model. "
        if fun<self.sol['fun']:
            hp,parameters_set=self.make_hp(theta,parameters)
            if self.modification:
                prefactor2=np.mean(co_Kinv*coef_re)-(np.mean(coef_re/np.sqrt(K_inv_diag))**2)
                hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)])))
                self.sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            else:
                self.sol['x']=theta.copy()
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol
    
    def copy(self):
        " Copy the objective function object. "
        return self.__class__(get_prior_mean=self.get_prior_mean,modification=self.modification)
    
    def __repr__(self):
        return "{}(get_prior_mean={},modification={})".format(self.__class__.__name__,self.get_prior_mean,self.modification)
    