import numpy as np
from .objectfunction import Object_functions
from ..hptrans import Variable_Transformation
from ..optimizers.local_opt import run_golden
from scipy.linalg import svd


class FactorizedLogLikelihoodSVD(Object_functions):
    
    def __init__(self,optimize=True,multiple_max=False,tol=1e-5,ngrid=50,maxiter=500,use_bounds=True,s=0.14):
        self.optimize=optimize
        self.multiple_max=multiple_max
        self.tol=tol
        self.ngrid=ngrid
        self.maxiter=maxiter
        self.use_bounds=use_bounds
        self.s=s
    
    def function(self,theta,TP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        hp,parameters_set=self.hp(theta,parameters)
        TP=self.update(TP,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(TP,X,Y,dis_m)
        noise,nlp=self.maximize_noise(TP,X,Y_p,parameters_set,parameters,prior,UTY,D,n_data)
        if jac==False:
            return nlp
        # Derivatives
        nlp_deriv=np.array([])
        D_n=D+np.exp(2*noise)
        hp=TP.get_hyperparameters()
        hp.update(noise=np.array([noise]).reshape(-1))
        TP.set_hyperparams(hp)
        KXX_inv=np.matmul(U/D_n,U.T)
        coef=np.matmul(KXX_inv,Y_p)
        ycoef=1+np.sum(UTY/D_n)/(2*TP.b)
        for para in parameters_set:
            K_deriv=TP.get_gradients(X,[para],KXX=KXX,dis_m=dis_m)[para]
            multiple_para=len(hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nlp_deriv=np.append(nlp_deriv,-0.5*((2*TP.a+n_data)/(2*TP.b))*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)/ycoef+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,prior,jac=True)
        return nlp,nlp_deriv 
    
    def get_eig(self,GP,X,Y,dis_m):
        " Calculate the SVD " 
        # Calculate the kernel with and without noise
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dis_m=dis_m)
        n_data=len(KXX)
        #KXX[range(n_data),range(n_data)]+=GP.get_correction(np.diag(KXX))
        # SVD
        U,D,Vt=svd(KXX)
        # Subtract the prior mean to the training target
        Y_p,GP=self.y_prior(X,Y,GP)
        VYUTY=np.matmul(Vt,Y_p).reshape(-1)**2
        return D,U,Y_p,VYUTY,KXX,n_data
    
    def get_eig_ll(self,noise,hp,parameters_set,parameters,prior,UTY,D,n_data,a,b):
        " Calculate log-likelihood from Eigendecomposition "
        D_n=D+np.exp(2*noise)
        nlp=0.5*np.sum(np.log(D_n))+0.5*(2*a+n_data)*np.log(1+np.sum(UTY/D_n)/(2*b))
        hp.update(dict(noise=np.array([noise]).reshape(-1)))
        return nlp-self.logpriors(hp,parameters_set,parameters,prior,jac=False)
    
    def maximize_noise(self,TP,X,Y,parameters_set,parameters,prior,UTY,D,n_data):
        " Find the maximum noise "
        noises=self.make_noise_list(TP,X,Y)
        args_ll=(TP.hp.copy(),parameters_set,parameters,prior,UTY,D,n_data,TP.a,TP.b)
        sol=run_golden(self.get_eig_ll,noises,maxiter=self.maxiter,tol=self.tol,optimize=self.optimize,multiple_max=self.multiple_max,args=args_ll)
        return sol['x'],sol['fun']

    def make_noise_list(self,TP,X,Y):
        " Make the list of noises in the variable transformation space " 
        hyper_var=Variable_Transformation().transf_para(['noise'],TP,X,Y,use_bounds=self.use_bounds,s=self.s)
        dl=np.finfo(float).eps
        noises=[np.linspace(0.0+dl,1.0-dl,self.ngrid)]
        return hyper_var.t_to_theta_lines(noises,['noise']).reshape(-1)
    
    def get_solution(self,sol,TP,parameters,X,Y,prior,jac=False,dis_m=None):
        " Get the solution of the optimization in terms of hyperparameters and TP "
        hp,parameters_set=self.hp(sol['x'],parameters)
        TP=self.update(TP,hp)
        D,U,Y_p,UTY,KXX,n_data=self.get_eig(TP,X,Y,dis_m)
        noise,nlp=self.maximize_noise(TP,X,Y_p,parameters_set,parameters,prior,UTY,D,n_data)
        hp.update(dict(noise=np.array([noise]).reshape(-1)))
        sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
        sol['hp']=hp.copy()
        sol['TP']=self.update(TP,hp)
        sol['nfev']+=1
        return sol
        