import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction

class LogLikelihood(ObjectiveFuction):
    """ Log-likelihood objective function as a function of the hyperparameters. """
    
    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        prefactor=model.hp['prefactor'][0]
        prefactor2=np.exp(-2.0*prefactor)
        nlp=0.5*prefactor2*np.matmul(Y_p.T,coef)+n_data*prefactor+np.sum(np.log(np.diagonal(L)))+0.5*n_data*np.log(2.0*np.pi)
        nlp=nlp.item(0)-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
        if jac:
            return nlp,self.derivative(hp,parameters_set,parameters,model,X,Y_p,KXX,L,low,coef,prefactor2,n_data,pdis,**kwargs)   
        return nlp
    
    def derivative(self,hp,parameters_set,parameters,model,X,Y_p,KXX,L,low,coef,prefactor2,n_data,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        nlp_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        for para in parameters_set:
            if para=='prefactor':
                nlp_deriv=np.append(nlp_deriv,-prefactor2*np.matmul(Y_p.T,coef)+n_data)
                continue
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv)
            nlp_deriv=np.append(nlp_deriv,-0.5*prefactor2*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)
        return nlp_deriv

