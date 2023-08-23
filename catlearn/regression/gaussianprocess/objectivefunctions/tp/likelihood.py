import numpy as np
from scipy.linalg import cho_solve
from ..objectivefunction import ObjectiveFuction

class LogLikelihood(ObjectiveFuction):
    """ Log-likelihood objective function as a function of the hyperparameters. """
    
    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        ycoef=1+(np.matmul(Y_p.T,coef).item(0)/(2*model.b))
        nlp=np.sum(np.log(np.diagonal(L)))+0.5*(2.0*model.a+n_data)*np.log(ycoef)
        nlp=nlp.item(0)-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
        if jac:
            return nlp,self.derivative(hp,parameters_set,parameters,model,X,Y_p,KXX,L,low,coef,ycoef,n_data,pdis,**kwargs)   
        return nlp
    
    def derivative(self,hp,parameters_set,parameters,model,X,Y_p,KXX,L,low,coef,ycoef,n_data,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        nlp_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        for para in parameters_set:
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv)
            nlp_deriv=np.append(nlp_deriv,-0.5*((model.a+0.5*n_data)/(model.b))*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)/ycoef+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)
        return nlp_deriv

