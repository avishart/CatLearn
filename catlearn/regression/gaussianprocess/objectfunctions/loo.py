import numpy as np
from scipy.linalg import cho_solve
from .objectfunction import Object_functions

class LOO(Object_functions):
    def __init__(self,modification=False):
        " The leave-one-out object function that is used to optimize the hyperparameters "
        self.modification=modification
    
    def function(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        GP=self.update(GP,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        KXX_inv,K_inv_diag,coef,co_Kinv=self.get_co_Kinv(L,low,n_data,coef)
        loo_v=np.mean(co_Kinv**2)
        loo_v=loo_v-self.logpriors(GP.hp.copy(),parameters_set,parameters,prior,jac=False)/n_data
        if jac==False:
            return loo_v
        # Derivatives
        loo_deriv=np.array([])
        hp=GP.get_hyperparameters()
        for para in parameters_set:
            if para=='prefactor':
                loo_deriv=np.append(loo_deriv,0.0*hp[para])
                continue
            K_deriv=GP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            if multiple_para:
                loo_d=2*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv),axis=1)
            else:
                loo_d=2*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv))
            loo_deriv=np.append(loo_deriv,loo_d)
        loo_deriv=loo_deriv-self.logpriors(hp,parameters_set,parameters,prior,jac=True)/n_data
        return loo_v,loo_deriv
    
    def get_co_Kinv(self,L,low,n_data,coef):
        " Get the analytic solution to the prefactor "
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        coef=coef.reshape(-1)
        co_Kinv=coef/K_inv_diag
        return KXX_inv,K_inv_diag,coef,co_Kinv
    
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
        if self.modification:
            hp.update(dict(prefactor=np.array([0.0])))
            GP=self.update(GP,hp)
            coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
            KXX_inv,K_inv_diag,coef,co_Kinv=self.get_co_Kinv(L,low,n_data,coef)
            prefactor2=np.mean(co_Kinv*coef)-(np.mean(coef/np.sqrt(K_inv_diag))**2)
            hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)])))
            sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            sol['nfev']+=1
        sol['hp']=hp.copy()
        sol['GP']=self.update(GP,hp)
        return sol
