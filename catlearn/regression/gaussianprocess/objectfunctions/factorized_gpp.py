import numpy as np
from .objectfunction import Object_functions
from ..hptrans import Variable_Transformation
from ..optimizers.local_opt import run_golden

class FactorizedGPP(Object_functions):
    
    def __init__(self,optimize=True,multiple_max=False,tol=1e-5,ngrid=50,maxiter=500,use_bounds=True,s=0.14):
        self.optimize=optimize
        self.multiple_max=multiple_max
        self.tol=tol
        self.ngrid=ngrid
        self.maxiter=maxiter
        self.use_bounds=use_bounds
        self.s=s
    
    def function(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        GP=self.update(GP,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(GP,X,Y,dis_m)
        noise,gpp_v=self.maximize_noise(GP,X,Y_p,parameters_set,parameters,prior,U,UTY,D,n_data)
        if jac==False:
            return gpp_v
        # Derivatives
        gpp_deriv=np.array([])
        D_n,UDn,K_inv_diag,coef,prefactor2=self.prefactor2(noise,U,D,UTY)
        KXX_inv=np.matmul(UDn,U.T)
        co_Kinv=coef/K_inv_diag
        hp=GP.hp.copy()
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))   
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
        gpp_deriv=gpp_deriv-self.logpriors(hp.copy(),parameters_set,parameters,prior,jac=True)/n_data
        return gpp_v,gpp_deriv
    
    def get_eig_ll(self,noise,hp,parameters_set,parameters,prior,U,UTY,D,n_data):
        " Calculate GPP from Eigendecomposition "
        D_n,UDn,K_inv_diag,coef,prefactor2=self.prefactor2(noise,U,D,UTY)
        gpp_v=1-np.mean(np.log(K_inv_diag))+np.log(prefactor2)+np.log(2*np.pi)
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        return gpp_v-self.logpriors(hp,parameters_set,parameters,prior,jac=False)/n_data
    
    def maximize_noise(self,GP,X,Y,parameters_set,parameters,prior,U,UTY,D,n_data):
        " Find the maximum noise "
        noises=self.make_noise_list(GP,X,Y)
        args_ll=(GP.hp.copy(),parameters_set,parameters,prior,U,UTY,D,n_data)
        sol=run_golden(self.get_eig_ll,noises,maxiter=self.maxiter,tol=self.tol,optimize=self.optimize,multiple_max=self.multiple_max,args=args_ll)
        return sol['x'],sol['fun']

    def make_noise_list(self,GP,X,Y):
        " Make the list of noises in the variable transformation space " 
        hyper_var=Variable_Transformation().transf_para(['noise'],GP,X,Y,use_bounds=self.use_bounds,s=self.s)
        dl=np.finfo(float).eps
        noises=[np.linspace(0.0+dl,1.0-dl,self.ngrid)]
        return hyper_var.t_to_theta_lines(noises,['noise']).reshape(-1)
    
    def prefactor2(self,noise,U,D,UTY):
        " Get the analytic solution to the prefactor "
        D_n=D+np.exp(2*noise)
        UDn=U/D_n
        coef=np.matmul(UDn,UTY).reshape(-1)
        K_inv_diag=np.einsum('ij,ji->i',UDn,U.T)
        prefactor2=np.mean((coef**2)/K_inv_diag)
        return D_n,UDn,K_inv_diag,coef,prefactor2
    
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
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(GP,X,Y,dis_m)
        noise,gpp_v=self.maximize_noise(GP,X,Y_p,parameters_set,parameters,prior,U,UTY,D,n_data)
        D_n,UDn,K_inv_diag,coef,prefactor2=self.prefactor2(noise,U,D,UTY)
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
        sol['hp']=hp.copy()
        sol['GP']=self.update(GP,hp)
        sol['nfev']+=1
        return sol
