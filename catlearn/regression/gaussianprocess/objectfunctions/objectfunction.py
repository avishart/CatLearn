import numpy as np
import copy
from scipy.linalg import cho_factor,cho_solve,eigh

class Object_functions:
    " The object function that is used to optimize the hyperparameters "
    def hp(self,theta,parameters):
        " Make hyperparameter dictionary from lists"
        theta,parameters=np.array(theta),np.array(parameters)
        parameters_set=sorted(set(parameters))
        hp={para_s:self.numeric_limits(theta[parameters==para_s]) for para_s in parameters_set}
        return hp,parameters_set
    
    def numeric_limits(self,array,dh=0.1*np.log(np.finfo(float).max)):
        " Replace hyperparameters if they are outside of the numeric limits in log-space "
        return np.where(-dh<array,np.where(array<dh,array,dh),-dh)
    
    def update(self,GP,hp):
        " Update GP "
        GP=copy.deepcopy(GP)
        GP.set_hyperparams(hp)
        return GP
    
    def kxx_corr(self,GP,X,dis_m=None):
        " Get covariance matrix with or without noise correction"
        # Calculate the kernel with and without noise
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dis_m=dis_m)
        n_data=len(KXX)
        KXX=self.add_correction(GP,KXX,n_data)
        return KXX,n_data
    
    def add_correction(self,GP,KXX,n_data):
        " Add noise correction to covariance matrix"
        if GP.correction:
            corr=GP.get_correction(np.diag(KXX))
            KXX[range(n_data),range(n_data)]+=corr
        return KXX
        
    def kxx_reg(self,GP,X,dis_m=None):
        " Get covariance matrix with regularization "
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dis_m=dis_m)
        KXX_n=GP.add_regularization(KXX,len(X),overwrite=False)
        return KXX_n,KXX,len(KXX)
        
    def y_prior(self,X,Y,GP):
        " Update prior and subtract target "
        Y_p=Y.copy()
        GP.prior.update(X,Y_p)
        Y_p=Y_p-GP.prior.get(X)
        if GP.use_derivatives:
            Y_p=Y_p.T.reshape(-1,1)
        return Y_p,GP
    
    def coef_cholesky(self,GP,X,Y,dis_m):
        " Calculate the coefficients by using Cholesky decomposition "
        # Calculate the kernel with and without noise
        KXX_n,KXX,n_data=self.kxx_reg(GP,X,dis_m=dis_m)
        # Cholesky decomposition
        L,low=cho_factor(KXX_n)
        # Subtract the prior mean to the training target
        Y_p,GP=self.y_prior(X,Y,GP)
        # Get the coefficients
        coef=cho_solve((L,low),Y_p,check_finite=False)
        return coef,L,low,Y_p,KXX,n_data

    def get_eig(self,GP,X,Y,dis_m):
        " Calculate the eigenvalues " 
        # Calculate the kernel with and without noise
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dis_m=dis_m)
        n_data=len(KXX)
        KXX[range(n_data),range(n_data)]+=GP.get_correction(np.diag(KXX))
        # Eigendecomposition
        D,U=eigh(KXX)
        # Subtract the prior mean to the training target
        Y_p,GP=self.y_prior(X,Y,GP)
        UTY=(np.matmul(U.T,Y_p)).reshape(-1)**2
        return D,U,Y_p,UTY,KXX,n_data
    
    def get_cinv(self,GP,X,Y,dis_m):
        " Get the inverse covariance matrix "
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(GP,X,Y,dis_m)
        cinv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        return coef,cinv,Y_p,KXX,n_data
    
    def logpriors(self,hp,parameters_set,parameters,prior=None,jac=False):
        " Log of the prior distribution value for the hyperparameters "
        if prior is None:
            return 0
        if not jac:
            lprior=0
            for para in set(hp.keys()):
                if para in prior.keys():
                    lprior+=np.sum([pr.ln_pdf(hp[para][p]) for p,pr in enumerate(prior[para])])
            return lprior
        lprior_deriv=np.array([])
        for para in parameters_set:
            if para in prior.keys():
                lprior_deriv=np.append(lprior_deriv,np.array([pr.ln_deriv(hp[para][p]) for p,pr in enumerate(prior[para])]))
            else:
                lprior_deriv=np.append(lprior_deriv,np.array([0]*parameters.count(para)))
        return lprior_deriv

    def get_K_inv_deriv(self,K_deriv,KXX_inv,multiple_para):
        " Get the diagonal elements of the matrix product of the inverse and derivative covariance matrix "
        if multiple_para:
            K_deriv_cho=np.array([np.einsum('ij,ji->',KXX_inv,K_d) for K_d in K_deriv])
        else:
            K_deriv_cho=np.einsum('ij,ji->',KXX_inv,K_deriv)
        return K_deriv_cho
    
    def get_solution(self,sol,GP,parameters,X,Y,prior,jac=False,dis_m=None):
        " Get the solution of the optimization in terms of hyperparameters and GP "
        hp,parameters_set=self.hp(sol['x'],parameters)
        sol['hp']=hp.copy()
        sol['GP']=self.update(GP,hp)
        return sol

    def function(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        " The function call that calculate the object function. "
        raise NotImplementedError()
