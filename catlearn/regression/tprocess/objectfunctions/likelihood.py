import numpy as np
from scipy.linalg import cho_solve
from .objectfunction import Object_functions


class LogLikelihood(Object_functions):
    
    def function(self,theta,TP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        TP=self.update(TP,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(TP,X,Y,dis_m)
        ycoef=1+(np.matmul(Y_p.T,coef).item(0)/(2*TP.b))
        nlp=np.sum(np.log(np.diagonal(L)))+0.5*(2*TP.a+n_data)*np.log(ycoef)
        nlp=nlp-self.logpriors(TP.hp.copy(),parameters_set,parameters,prior,jac=False)
        if jac==False:
            return nlp
        # Derivatives
        nlp_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        for para in parameters_set:
            K_deriv=TP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(TP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nlp_deriv=np.append(nlp_deriv,-0.5*((2*TP.a+n_data)/(2*TP.b))*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)/ycoef+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(TP.hp.copy(),parameters_set,parameters,prior,jac=True)
        return nlp,nlp_deriv  
