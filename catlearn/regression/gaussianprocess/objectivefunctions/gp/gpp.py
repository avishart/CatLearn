import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction

class GPP(ObjectiveFuction):
    """ The Geissers surrogate predictive probability objective function that is used to optimize the hyperparameters. """

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        KXX_inv,K_inv_diag,coef_re,co_Kinv,prefactor2=self.get_prefactor2(L,low,n_data,coef)
        gpp_v=1.0-np.mean(np.log(K_inv_diag))+np.log(prefactor2)+np.log(2.0*np.pi)
        gpp_v=gpp_v-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)/n_data
        if jac:
            deriv=self.derivative(hp,parameters_set,parameters,model,X,KXX,KXX_inv,K_inv_diag,coef_re,co_Kinv,prefactor2,n_data,pdis,**kwargs)  
            self.update_solution(gpp_v,theta,parameters,model,jac=jac,deriv=deriv,prefactor2=prefactor2)
            return gpp_v,deriv
        self.update_solution(gpp_v,theta,parameters,model,jac=jac,prefactor2=prefactor2)
        return gpp_v
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,KXX_inv,K_inv_diag,coef_re,co_Kinv,prefactor2,n_data,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        gpp_deriv=np.array([])
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1)))  
        for para in parameters_set:
            if para=='prefactor':
                gpp_deriv=np.append(gpp_deriv,np.zeros((len(hp[para]))))
                continue
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef_re)
            gpp_d=(np.mean(co_Kinv*(2.0*r_j+co_Kinv*s_j),axis=-1)/prefactor2)+np.mean(s_j/K_inv_diag,axis=-1)
            gpp_deriv=np.append(gpp_deriv,gpp_d)
        gpp_deriv=gpp_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)/n_data
        return gpp_deriv
    
    def get_prefactor2(self,L,low,n_data,coef):
        " Get the analytic solution to the prefactor "
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        coef_re=coef.reshape(-1)
        co_Kinv=coef_re/K_inv_diag
        prefactor2=np.mean(co_Kinv*coef_re)
        return KXX_inv,K_inv_diag,coef_re,co_Kinv,prefactor2
    
    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        r_j=np.einsum('ji,di->dj',KXX_inv,np.matmul(K_deriv,-coef))
        s_j=np.einsum('ji,dji->di',KXX_inv,np.matmul(K_deriv,KXX_inv))
        return r_j,s_j
    
    def update_solution(self,fun,theta,parameters,model,jac=False,deriv=None,prefactor2=None,**kwargs):
        " Update the solution of the optimization in terms of hyperparameters and model. "
        if fun<self.sol['fun']:
            hp,parameters_set=self.make_hp(theta,parameters)
            hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)])))
            self.sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol
