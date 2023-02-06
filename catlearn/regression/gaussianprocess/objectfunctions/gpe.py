import numpy as np
from scipy.linalg import cho_solve
from .objectfunction import Object_functions

class GPE(Object_functions):
    
    def function(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        GP=self.update(GP,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        KXX_inv,K_inv_diag_rev,coef,co_Kinv,prefactor2=self.get_co_Kinv(L,low,n_data,coef,GP.hp)
        gpe_v=np.mean(co_Kinv**2)+prefactor2*np.mean(K_inv_diag_rev)
        gpe_v=gpe_v-self.logpriors(GP.hp.copy(),parameters_set,parameters,prior,jac=False)/n_data
        if jac==False:
            return gpe_v
        # Derivatives
        gpe_deriv=np.array([])
        hp=GP.get_hyperparameters()
        for para in parameters_set:
            if para=='prefactor':
                gpe_deriv=np.append(gpe_deriv,2.0*prefactor2*np.mean(K_inv_diag_rev))
                continue
            K_deriv=GP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            if multiple_para:
                gpe_d=2*np.mean((co_Kinv*K_inv_diag_rev)*(r_j+s_j*co_Kinv),axis=1)+prefactor2*np.mean(s_j*(K_inv_diag_rev*K_inv_diag_rev),axis=1)
            else:
                gpe_d=2*np.mean((co_Kinv*K_inv_diag_rev)*(r_j+s_j*co_Kinv))+prefactor2*np.mean(s_j*(K_inv_diag_rev*K_inv_diag_rev))
            gpe_deriv=np.append(gpe_deriv,gpe_d)
        gpe_deriv=gpe_deriv-self.logpriors(hp,parameters_set,parameters,prior,jac=True)/n_data
        return gpe_v,gpe_deriv
    
    def get_co_Kinv(self,L,low,n_data,coef,hp):
        " Get the analytic solution to the prefactor "
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag_rev=1.0/np.diag(KXX_inv)
        coef=coef.reshape(-1)
        co_Kinv=coef*K_inv_diag_rev
        prefactor2=np.exp(2*hp['prefactor'].item(0))
        return KXX_inv,K_inv_diag_rev,coef,co_Kinv,prefactor2
    
    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef,multiple_para):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        if multiple_para:
            r_j=np.array([-np.matmul(np.matmul(KXX_inv,K_d),coef) for K_d in K_deriv])
            s_j=np.array([np.einsum('ij,ji->i',np.matmul(KXX_inv.T,K_d),KXX_inv) for K_d in K_deriv])
        else:
            r_j=-np.matmul(np.matmul(KXX_inv,K_deriv),coef)
            s_j=np.einsum('ij,ji->i',np.matmul(KXX_inv.T,K_deriv),KXX_inv)
        return r_j,s_j
