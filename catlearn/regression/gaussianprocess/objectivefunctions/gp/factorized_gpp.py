import numpy as np
from .factorized_likelihood import FactorizedLogLikelihood
from ...optimizers.local_opt import run_golden,fine_grid_search

class FactorizedGPP(FactorizedLogLikelihood):
    """ The factorized Geissers surrogate predictive probability objective function that is used to optimize the hyperparameters. 
        The prefactor hyperparameter is determined from an analytical expression. 
        An eigendecomposition is performed to get the eigenvalues. 
        The noise hyperparameter can be searched from a single eigendecomposition for each length-scale hyperparameter. 
    """

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(model,X,Y)
        noise,gpp_v=self.maximize_noise(model,X,Y_p,hp.copy(),parameters_set,parameters,pdis,U,UTY,D,n_data)
        if jac:
            deriv=self.derivative(hp,parameters_set,parameters,model,X,KXX,D,U,Y_p,UTY,noise,pdis,n_data,**kwargs)
            self.update_solution(gpp_v,theta,parameters,model,jac=jac,deriv=deriv,noise=noise,UTY=UTY,U=U,D=D,n_data=n_data)
            return gpp_v,deriv
        self.update_solution(gpp_v,theta,parameters,model,jac=jac,noise=noise,UTY=UTY,U=U,D=D,n_data=n_data)
        return gpp_v
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,D,U,Y_p,UTY,noise,pdis,n_data,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        gpp_deriv=np.array([])
        D_n=D+np.exp(2*noise)
        UDn=U/D_n
        KXX_inv=np.matmul(UDn,U.T)
        K_inv_diag=np.diag(KXX_inv)
        prefactor2=np.mean((np.matmul(UDn,UTY).reshape(-1)**2)/K_inv_diag)
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        coef_re=np.matmul(KXX_inv,Y_p).reshape(-1)
        co_Kinv=coef_re/K_inv_diag
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
    
    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        r_j=np.einsum('ji,di->dj',KXX_inv,np.matmul(K_deriv,-coef))
        s_j=np.einsum('ji,dji->di',KXX_inv,np.matmul(K_deriv,KXX_inv))
        return r_j,s_j
    
    def get_eig_ll(self,noise,hp,parameters_set,parameters,pdis,U,UTY,D,n_data,**kwargs):
        " Calculate GPP from Eigendecomposition for a noise value. "
        D_n=D+np.exp(2*noise)
        UDn=U/D_n
        K_inv_diag=np.einsum('ij,ji->i',UDn,U.T)
        prefactor2=np.log(np.mean((np.matmul(UDn,UTY).reshape(-1)**2)/K_inv_diag))
        gpp_v=1-np.mean(np.log(K_inv_diag))+prefactor2+np.log(2*np.pi)
        hp.update(dict(prefactor=np.array([0.5*prefactor2]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        return gpp_v-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)/n_data
    
    def get_all_eig_ll(self,noises,fun,hp,parameters_set,parameters,pdis,U,UTY,D,n_data,**kwargs):
        " Calculate GPP from Eigendecompositions for all noise values from the list. "
        D_n=D+np.exp(2*noises)
        UDn=U/D_n[:,None,:]
        K_inv_diag=np.einsum('dij,ji->di',UDn,U.T,optimize=True)
        prefactor2=np.log(np.mean((np.matmul(UDn,UTY).reshape((len(noises),n_data))**2)/K_inv_diag,axis=1))
        gpp_v=1-np.mean(np.log(K_inv_diag),axis=1)+prefactor2+np.log(2*np.pi)
        hp.update(dict(prefactor=0.5*prefactor2.reshape(-1,1),noise=noises))
        return gpp_v-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)/n_data
    
    def maximize_noise_golden(self,model,X,Y,hp,parameters_set,parameters,pdis,U,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method combined with the golden section search method for local optimization. "
        noises=self.make_noise_list(model,X,Y)
        args_ll=(hp,parameters_set,parameters,pdis,U,UTY,D,n_data)
        sol=run_golden(self.get_eig_ll,noises,fun_list=self.get_all_eig_ll,args=args_ll,**self.method_kwargs)
        return sol['x'],sol['fun']
    
    def maximize_noise_finegrid(self,model,X,Y,hp,parameters_set,parameters,pdis,U,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method combined with a finer grid method for local optimization. "
        noises=self.make_noise_list(model,X,Y)
        args_ll=(hp,parameters_set,parameters,pdis,U,UTY,D,n_data)
        sol=fine_grid_search(self.get_eig_ll,noises,fun_list=self.get_all_eig_ll,args=args_ll,**self.method_kwargs)
        return sol['x'],sol['fun']
    
    def maximize_noise_grid(self,model,X,Y,hp,parameters_set,parameters,pdis,U,UTY,D,n_data,**kwargs):
        " Find the maximum noise with a grid method. "
        noises=self.make_noise_list(model,X,Y)
        # Calculate function values for line coordinates
        f_list=self.get_all_eig_ll(noises,self.get_eig_ll,hp,parameters_set,parameters,pdis,U,UTY,D,n_data)
        # Find the minimum value
        i_min=np.nanargmin(f_list)
        return noises[i_min],f_list[i_min]

    def update_solution(self,fun,theta,parameters,model,jac=False,deriv=None,noise=None,UTY=None,U=None,D=None,n_data=None,**kwargs):
        " Update the solution of the optimization in terms of hyperparameters and model. "
        if fun<self.sol['fun']:
            hp,parameters_set=self.make_hp(theta,parameters)
            D_n=D+np.exp(2.0*noise)
            UDn=U/D_n
            K_inv_diag=np.einsum('ij,ji->i',UDn,U.T)
            prefactor2=np.mean((np.matmul(UDn,UTY).reshape(-1)**2)/K_inv_diag)
            if self.modification:
                prefactor2=(n_data/(n_data-len(theta)))*prefactor2 if n_data-len(theta)>0 else prefactor2
            hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]),noise=np.array([noise]).reshape(-1)))
            self.sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol
