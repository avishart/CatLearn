import numpy as np
from scipy.linalg import cho_solve
from .objectfunction import Object_functions

class MaximumLogLikelihood(Object_functions):
    
    def function(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        GP=self.update(GP,hp)
        #GP.set_hyperparams(hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        prefactor2=np.matmul(Y_p.T,coef).item(0)/n_data
        nlp=0.5*n_data*(1+np.log(2*np.pi)+np.log(prefactor2))+np.sum(np.log(np.diagonal(L)))
        nlp=nlp.item(0)-self.logpriors(GP.hp.copy(),parameters_set,parameters,prior,jac=False)
        if jac==False:
            return nlp
        # Derivatives
        nlp_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        for para in parameters_set:
            if para=='prefactor':
                nlp_deriv=np.append(nlp_deriv,0*GP.hp[para])
                continue
            K_deriv=GP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nlp_deriv=np.append(nlp_deriv,-(0.5*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1))/prefactor2+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(GP.hp.copy(),parameters_set,parameters,prior,jac=True)
        return nlp,nlp_deriv    
    
    def get_solution(self,sol,GP,parameters,X,Y,prior,jac=False,dis_m=None):
        " Get the solution of the optimization in terms of hyperparameters and GP "
        hp,parameters_set=self.hp(sol['x'],parameters)
        GP=self.update(GP,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        prefactor2=np.matmul(Y_p.T,coef).item(0)/n_data
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)])))
        sol['hp']=hp.copy()
        sol['GP']=self.update(GP,hp)
        sol['nfev']+=1
        return sol