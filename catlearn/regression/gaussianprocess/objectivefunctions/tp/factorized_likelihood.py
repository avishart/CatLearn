import numpy as np
from ..gp.factorized_likelihood import FactorizedLogLikelihood
from ...optimizers.local_opt import run_golden,fine_grid_search

class FactorizedLogLikelihood(FactorizedLogLikelihood):

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(model,X,Y)
        noise,nlp=self.maximize_noise(model,X,Y_p,hp.copy(),parameters_set,parameters,pdis,UTY,D,n_data)
        if jac:
            deriv=self.derivative(hp,parameters_set,parameters,model,X,KXX,D,U,Y_p,UTY,noise,pdis,n_data,**kwargs)
            self.update_solution(nlp,theta,parameters,model,jac=jac,deriv=deriv,noise=noise)
            return nlp,deriv
        self.update_solution(nlp,theta,parameters,model,jac=jac,noise=noise)
        return nlp
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,D,U,Y_p,UTY,noise,pdis,n_data,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        nlp_deriv=np.array([])
        D_n=D+np.exp(2*noise)
        hp.update(dict(noise=np.array([noise]).reshape(-1)))
        KXX_inv=np.matmul(U/D_n,U.T)
        coef=np.matmul(KXX_inv,Y_p)
        ycoef=1+np.sum(UTY/D_n)/(2*model.b)
        for para in parameters_set:
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv)
            nlp_deriv=np.append(nlp_deriv,-0.5*((model.a+0.5*n_data)/(model.b))*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)/ycoef+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)
        return nlp_deriv
    
    def get_eig_ll(self,noise,hp,parameters_set,parameters,pdis,UTY,D,n_data,a,b,**kwargs):
        " Calculate log-likelihood from Eigendecomposition for a noise value. "
        D_n=D+np.exp(2*noise)
        nlp=0.5*np.sum(np.log(D_n))+0.5*(2.0*a+n_data)*np.log(1.0+np.sum(UTY/D_n)/(2.0*b))
        hp.update(dict(noise=np.array([noise]).reshape(-1)))
        return nlp-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
    
    def get_all_eig_ll(self,noises,fun,hp,parameters_set,parameters,pdis,UTY,D,n_data,a,b,**kwargs):
        " Calculate log-likelihood from Eigendecompositions for all noise values from the list. "
        D_n=D+np.exp(2*noises)
        nlp=0.5*np.sum(np.log(D_n),axis=1)+0.5*(2.0*a+n_data)*np.log(1.0+np.sum(UTY/D_n,axis=1)/(2.0*b))
        hp.update(dict(noise=noises))
        return nlp-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
    
    def maximize_noise_golden(self,model,X,Y,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method combined with the golden section search method for local optimization. "
        noises=self.make_noise_list(model,X,Y)
        args_ll=(hp,parameters_set,parameters,pdis,UTY,D,n_data,model.a,model.b)
        sol=run_golden(self.get_eig_ll,noises,fun_list=self.get_all_eig_ll,args=args_ll,**self.method_kwargs)
        return sol['x'],sol['fun']
    
    def maximize_noise_finegrid(self,model,X,Y,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method combined with a finer grid method for local optimization. "
        noises=self.make_noise_list(model,X,Y)
        args_ll=(hp,parameters_set,parameters,pdis,UTY,D,n_data,model.a,model.b)
        sol=fine_grid_search(self.get_eig_ll,noises,fun_list=self.get_all_eig_ll,args=args_ll,**self.method_kwargs)
        return sol['x'],sol['fun']
    
    def maximize_noise_grid(self,model,X,Y,hp,parameters_set,parameters,pdis,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method. "
        noises=self.make_noise_list(model,X,Y)
        # Calculate function values for line coordinates
        f_list=self.get_all_eig_ll(noises,self.get_eig_ll,hp,parameters_set,parameters,pdis,UTY,D,n_data,model.a,model.b)
        # Find the minimum value
        i_min=np.nanargmin(f_list)
        return noises[i_min],f_list[i_min]

    def update_solution(self,fun,theta,parameters,model,jac=False,deriv=None,noise=None,**kwargs):
        " Update the solution of the optimization in terms of hyperparameters and model. "
        if fun<self.sol['fun']:
            hp,parameters_set=self.make_hp(theta,parameters)
            hp.update(dict(noise=np.array([noise]).reshape(-1)))
            self.sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol

    def __repr__(self):
        return "{}(get_prior_mean={},modification={},ngrid={},hptrans={},use_bounds={},s={},noise_method={finegrid})".format(self.__class__.__name__,self.get_prior_mean,self.modification,self.ngrid,self.hptran,self.use_bounds,self.s,self.noise_method)
