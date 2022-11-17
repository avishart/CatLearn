import numpy as np
from scipy.linalg import cho_solve
from .objectfunction import Object_functions

class GPP(Object_functions):
    
    def function(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        GP=self.update(GP,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        KXX_inv,K_inv_diag,coef,co_Kinv,prefactor2=self.prefactor2(L,low,n_data,coef)
        gpp_v=1-np.mean(np.log(K_inv_diag))+np.log(prefactor2)+np.log(2*np.pi)
        gpp_v=gpp_v-self.logpriors(GP.hp.copy(),parameters_set,parameters,prior,jac=False)/n_data
        if jac==False:
            return gpp_v
        # Derivatives
        gpp_deriv=np.array([])
        hp=GP.hp.copy()
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1)))  
        for para in parameters_set:
            if para=='prefactor':
                gpp_deriv=np.append(gpp_deriv,0.0*hp[para])
                continue
            K_deriv=GP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            if multiple_para:
                gpp_d=(np.mean(co_Kinv*(2*r_j+co_Kinv*s_j),axis=1)/prefactor2)+np.mean(s_j/K_inv_diag,axis=1)
            else:
                gpp_d=(np.mean(co_Kinv*(2*r_j+co_Kinv*s_j))/prefactor2)+np.mean(s_j/K_inv_diag)
            gpp_deriv=np.append(gpp_deriv,gpp_d)
        gpp_deriv=gpp_deriv-self.logpriors(hp,parameters_set,parameters,prior,jac=True)/n_data
        return gpp_v,gpp_deriv
    
    def prefactor2(self,L,low,n_data,coef):
        " Get the analytic solution to the prefactor "
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        coef=coef.reshape(-1)
        co_Kinv=coef/K_inv_diag
        prefactor2=np.mean(co_Kinv*coef)
        return KXX_inv,K_inv_diag,coef,co_Kinv,prefactor2
    
    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef,multiple_para):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        if multiple_para:
            r_j=np.array([-np.matmul(np.matmul(KXX_inv,K_d),coef) for K_d in K_deriv])
            s_j=np.array([np.einsum('ij,ji->i',np.matmul(KXX_inv.T,K_d),KXX_inv) for K_d in K_deriv])
        else:
            r_j=-np.matmul(np.matmul(KXX_inv,K_deriv),coef)
            s_j=np.einsum('ij,ji->i',np.matmul(KXX_inv.T,K_deriv),KXX_inv)
        return r_j,s_j
    
    def get_solution(self,sol,GP,parameters,X,Y,prior,jac=False,dis_m=None):
        " Get the solution of the optimization in terms of hyperparameters and GP "
        hp,parameters_set=self.hp(sol['x'],parameters)
        GP=self.update(GP,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        KXX_inv,K_inv_diag,coef,co_Kinv,prefactor2=self.prefactor2(L,low,n_data,coef)
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)])))
        sol['hp']=hp.copy()
        sol['GP']=self.update(GP,hp)
        sol['nfev']+=1
        return sol
