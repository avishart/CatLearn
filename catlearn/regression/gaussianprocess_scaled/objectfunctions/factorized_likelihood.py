import numpy as np
from .objectfunction import Object_functions
from ..hptrans import Variable_Transformation
from ..optimizers.local_opt import run_golden


class FactorizedLogLikelihood(Object_functions):
    
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
        noise,nlp=self.maximize_noise(GP,X,Y_p,parameters_set,parameters,prior,UTY,D,n_data)
        if jac==False:
            return nlp
        # Derivatives
        nlp_deriv=np.array([])
        D_n=D+np.exp(2*noise)
        prefactor2=np.mean(UTY/D_n)
        hp=GP.hp.copy()
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))  
        KXX_inv=np.matmul(U/D_n,U.T)
        coef=np.matmul(KXX_inv,Y_p)
        for para in parameters_set:
            if para in ['prefactor']:
                nlp_deriv=np.append(nlp_deriv,0.0*GP.hp[para])
                continue
            K_deriv=GP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            Kdc=np.matmul(K_deriv,coef)
            if para=='length':
                Y_t=self.Y_l_transform(X,Y_p)
                nlp_deriv=np.append(nlp_deriv,-0.5*(-2*np.matmul(Y_t.T,coef)+np.matmul(coef.T,Kdc))/prefactor2.reshape(-1)+0.5*K_deriv_cho)
            else:
                nlp_deriv=np.append(nlp_deriv,-0.5*np.matmul(coef.T,Kdc).reshape(-1)/prefactor2+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,prior,jac=True)
        return nlp,nlp_deriv
    
    def get_eig_ll(self,noise,hp,parameters_set,parameters,prior,UTY,D,n_data):
        " Calculate log-likelihood from Eigendecomposition "
        D_n=D+np.exp(2*noise)
        prefactor2=np.mean(UTY/D_n)
        nlp=0.5*n_data*(np.log(prefactor2)+1+np.log(2*np.pi))+0.5*np.sum(np.log(D_n))
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        return nlp-self.logpriors(hp,parameters_set,parameters,prior,jac=False)
    
    def maximize_noise(self,GP,X,Y,parameters_set,parameters,prior,UTY,D,n_data):
        " Find the maximum noise "
        noises=self.make_noise_list(GP,X,Y)
        args_ll=(GP.hp.copy(),parameters_set,parameters,prior,UTY,D,n_data)
        sol=run_golden(self.get_eig_ll,noises,maxiter=self.maxiter,tol=self.tol,optimize=self.optimize,multiple_max=self.multiple_max,args=args_ll)
        return sol['x'],sol['fun']

    def make_noise_list(self,GP,X,Y):
        " Make the list of noises in the variable transformation space " 
        hyper_var=Variable_Transformation().transf_para(['noise'],GP,X,Y,use_bounds=self.use_bounds,s=self.s)
        dl=np.finfo(float).eps
        noises=[np.linspace(0.0+dl,1.0-dl,self.ngrid)]
        return hyper_var.t_to_theta_lines(noises,['noise']).reshape(-1)
    
    def get_solution(self,sol,GP,parameters,X,Y,prior,jac=False,dis_m=None):
        " Get the solution of the optimization in terms of hyperparameters and GP "
        hp,parameters_set=self.hp(sol['x'],parameters)
        GP=self.update(GP,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(GP,X,Y,dis_m)
        noise,nlp=self.maximize_noise(GP,X,Y_p,parameters_set,parameters,prior,UTY,D,n_data)
        D_n=D+np.exp(2*noise)
        prefactor2=np.mean(UTY/D_n)
        hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)]).reshape(-1),noise=np.array([noise]).reshape(-1)))
        sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
        sol['hp']=hp.copy()
        sol['GP']=self.update(GP,hp)
        sol['nfev']+=1
        return sol
