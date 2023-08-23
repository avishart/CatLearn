import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction

class GPE(ObjectiveFuction):
    """ The Geissers predictive mean square error objective function that is used to optimize the hyperparameters. """
    
    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        KXX_inv,K_inv_diag_rev,coef_re,co_Kinv=self.get_co_Kinv(L,low,n_data,coef)
        prefactor2=np.exp(2*hp['prefactor'][0])
        gpe_v=np.mean(co_Kinv**2)+prefactor2*np.mean(K_inv_diag_rev)
        gpe_v=gpe_v-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)/n_data
        if jac:
            return gpe_v,self.derivative(hp,parameters_set,parameters,model,X,KXX,KXX_inv,K_inv_diag_rev,coef_re,co_Kinv,prefactor2,n_data,pdis,**kwargs)   
        return gpe_v
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,KXX_inv,K_inv_diag_rev,coef_re,co_Kinv,prefactor2,n_data,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        gpe_deriv=np.array([])
        for para in parameters_set:
            if para=='prefactor':
                gpe_deriv=np.append(gpe_deriv,2.0*prefactor2*np.mean(K_inv_diag_rev))
                continue
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef_re)
            gpe_d=2*np.mean((co_Kinv*K_inv_diag_rev)*(r_j+s_j*co_Kinv),axis=-1)+prefactor2*np.mean(s_j*(K_inv_diag_rev*K_inv_diag_rev),axis=-1)
            gpe_deriv=np.append(gpe_deriv,gpe_d)
        gpe_deriv=gpe_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)/n_data
        return gpe_deriv
    
    def get_co_Kinv(self,L,low,n_data,coef):
        " Get the analytic solution to the prefactor "
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag_rev=1.0/np.diag(KXX_inv)
        coef_re=coef.reshape(-1)
        co_Kinv=coef_re*K_inv_diag_rev
        return KXX_inv,K_inv_diag_rev,coef_re,co_Kinv
    
    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        r_j=np.einsum('ji,di->dj',KXX_inv,np.matmul(K_deriv,-coef))
        s_j=np.einsum('ji,dji->di',KXX_inv,np.matmul(K_deriv,KXX_inv))
        return r_j,s_j
