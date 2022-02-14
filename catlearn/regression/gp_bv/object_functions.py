import numpy as np
import copy
from scipy.linalg import cho_factor,cho_solve


class Object_functions:
    def __init__(self,fun='nmll',log=True,cost_args={}):
        " The object function used for optimization of hyperparameters "
        self.fun_name=fun.lower()
        self.log=log
        self.cost_args={'Qu':1,'Wp':1,'Wu':1,'combi':'sum'}
        self.cost_args.update(cost_args)
        self.fun_choice(self.fun_name,log)

    def fun_choice(self,fun,log):
        " Get the object function specified by fun "
        # What object function to use
        fun_f={'nmll':self.nmll,'nml':self.nml,'loo':self.loo,'gpe':self.gpe,'gpp':self.gpp,'nmlp':self.nmlp,\
               'nmlp+nmll':self.nmlp,'mp':self.mp,'cost':self.cost,'mnll':self.mnll,'mnlp':self.mnlp}
        fun_f=fun_f[fun.lower()]
        # A wrapper function is used if hyperparameters are in log space
        if log:
            self.func=fun_f
            fun_f=self.log_wrapper
        self.fun=fun_f
        pass

    def nmll(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Negative mean log likelihood"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        nmll=(np.matmul(Y.T,coef)+2*np.sum(np.log(np.diagonal(L)))).item(0)/n_data+np.log(2*np.pi)
        if jac==False:
            return nmll
        # Derivatives
        nmll_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nmll_deriv=np.append(nmll_deriv,-sign_t[para]*(np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)-K_deriv_cho)/n_data)
        return nmll,nmll_deriv

    def nml(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Negative mean likelihood"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        nmml=(np.matmul(Y.T,coef)+2*np.sum(np.log(np.diagonal(L)))).item(0)/n_data+np.log(2*np.pi)
        nmml=-np.exp(-0.5*nmml)
        if jac==False:
            return nmml
        # Derivatives
        nmml_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nmml_deriv=np.append(nmml_deriv,sign_t[para]*(np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)-K_deriv_cho)/n_data)
        return nmml,(0.5*nmml)*nmml_deriv

    def loo(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Leave-one-out estimation"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        coef=coef.reshape(-1)
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        co_Kinv=coef/K_inv_diag
        loo_v=np.mean(co_Kinv**2)
        if jac==False:
            return loo_v
        # Derivatives
        loo_deriv=np.array([])
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            if multiple_para:
                loo_d=2*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv),axis=1)
            else:
                loo_d=2*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv))
            loo_deriv=np.append(loo_deriv,sign_t[para]*loo_d)
        return loo_v,loo_deriv
    
    def gpe(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Geissers predictive mean square error with derivative for optimization"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        coef=coef.reshape(-1)
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag_rev=1/np.diag(KXX_inv)
        co_Kinv=coef*K_inv_diag_rev
        gpe_v=np.mean(co_Kinv**2+K_inv_diag_rev)
        if jac==False:
            return gpe_v
        # Derivatives
        gpe_deriv=np.array([])
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            if multiple_para:
                gpp_d=np.mean(K_inv_diag_rev*(s_j*(K_inv_diag_rev+2*co_Kinv**2)+2*co_Kinv*r_j),axis=1)
            else:
                gpp_d=np.mean(K_inv_diag_rev*(s_j*(K_inv_diag_rev+2*co_Kinv**2)+2*co_Kinv*r_j))
            gpe_deriv=np.append(gpe_deriv,sign_t[para]*gpp_d)
        return gpe_v,gpe_deriv

    def gpp(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Geissers surrogate predictive probability with derivative for optimization"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        coef=coef.reshape(-1)
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        gpp_v=np.mean((coef**2)/K_inv_diag-np.log(K_inv_diag))+np.log(2*np.pi)
        if jac==False:
            return gpp_v
        # Derivatives
        gpp_deriv=np.array([])
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            if multiple_para:
                gpp_d=np.mean((s_j/K_inv_diag)*(1+((coef**2)/K_inv_diag))+(2*r_j*coef)/K_inv_diag,axis=1)
            else:
                gpp_d=np.mean((s_j/K_inv_diag)*(1+((coef**2)/K_inv_diag))+(2*r_j*coef)/K_inv_diag)
            gpp_deriv=np.append(gpp_deriv,sign_t[para]*gpp_d)
        return gpp_v,gpp_deriv

    def nmlp(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Negative mean log posterior"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        nmll=(np.matmul(Y.T,coef)+2*np.sum(np.log(np.diagonal(L)))).item(0)/n_data+np.log(2*np.pi)
        if prior is not None:
            nmll+=-2*np.sum([np.sum([pr.ln_pdf(GP.hp[para][p]) for p,pr in enumerate(prior[para])]) for para in prior.keys()])/n_data
        # Derivatives
        if jac==False:
            return nmll
        nmll_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nmll_d=-(np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)-K_deriv_cho)/n_data
            if prior is not None:
                nmll_d+=-2*np.array([pr.ln_deriv(GP.hp[para][p]) for p,pr in enumerate(prior[para])])/n_data
            nmll_deriv=np.append(nmll_deriv,sign_t[para]*nmll_d)
        return nmll,nmll_deriv

    def mp(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Marginal posterior"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        nml=(np.matmul(Y.T,coef)+2*np.sum(np.log(np.diagonal(L)))).item(0)/n_data+np.log(2*np.pi)
        if prior is not None:
            nml+=-2*np.sum([np.sum([pr.ln_pdf(GP.hp[para][p]) for p,pr in enumerate(prior[para])]) for para in prior.keys()])/n_data
        nml=np.exp(-0.5*nml)
        # Derivatives
        if jac==False:
            return nml
        nml_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)         
            nml_d=-(np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)-K_deriv_cho)/n_data
            if prior is not None:
                nml_d+=-2*np.array([pr.ln_deriv(GP.hp[para][p]) for p,pr in enumerate(prior[para])])/n_data
            nml_deriv=np.append(nml_deriv,sign_t[para]*nml_d)
        return nml,(-0.5*nml)*nml_deriv

    def cost(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Cost function combined by a prediction and an uncertainty contributions"
        std_p=10/np.sqrt(np.mean((Y[:,0]-GP.prior.get(X))**2))
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        coef=coef.reshape(-1)
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_inv_diag=np.diag(KXX_inv)
        co_Kinv=coef/K_inv_diag
        n_var=(n_data-len(theta)) if (n_data-len(theta))>0 else 1
        eta_r=np.sum(co_Kinv*coef)/n_var
        qp,qu=self.cost_p_unc(co_Kinv,K_inv_diag,eta_r,n_var,std_p,jac=False)
        cost_value=self.cost_combi(qp,qu,jac=False)
        if not jac:
            return cost_value
        # Derivatives
        cost_deriv=np.array([])
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                K_deriv=self.get_derivative_correction(K_deriv,GP.hp,sign_t,n_data)
            multiple_para=len(GP.hp[para])>1
            r_j,s_j=self.get_r_s_derivatives(K_deriv,KXX_inv,coef,multiple_para)
            qp_d,qu_d=self.cost_p_unc(co_Kinv,K_inv_diag,eta_r,n_var,std_p,jac,multiple_para,r_j,s_j,qp,qu)
            cost_d=self.cost_combi(qp,qu,jac,cost_value,qp_d,qu_d)
            cost_deriv=np.append(cost_deriv,sign_t[para]*cost_d)
        return cost_value,cost_deriv

    def cost_pred(self,co_Kinv,K_inv_diag,std_p,jac=False,qp=False,multiple_para=False,r_j=None,s_j=None):
        "Predective cost function"
        if not jac:
            return std_p*np.sqrt(np.mean(co_Kinv**2))
        if multiple_para:
            return (std_p**2)*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv),axis=1)/qp
        return (std_p**2)*np.mean((co_Kinv/K_inv_diag)*(r_j+s_j*co_Kinv))/qp

    def cost_unc1(self,eta_r,n_var,jac=False,qu=False,multiple_para=False,co_Kinv=None,rj=None,s_j=None):
        "First uncertainty cost function"
        if not jac:
            return eta_r+(1/eta_r)-2
        if multiple_para:
            return np.sum(2*co_Kinv*rj+(co_Kinv**2)*s_j,axis=1)*(1-1/(eta_r**2))/n_var
        return np.sum(2*co_Kinv*rj+(co_Kinv**2)*s_j)*(1-1/(eta_r**2))/n_var

    def cost_unc2(self,eta_r,n_var,jac=False,qu=False,multiple_para=False,co_Kinv=None,rj=None,s_j=None):
        "Second uncertainty cost function"
        if not jac:
            return np.log(eta_r)**2
        if multiple_para:
            return 2*np.sqrt(qu)*np.sum(2*co_Kinv*rj+(co_Kinv**2)*s_j,axis=1)/(eta_r*n_var)
        return 2*np.sqrt(qu)*np.sum(2*co_Kinv*rj+(co_Kinv**2)*s_j)/(eta_r*n_var)

    def cost_p_unc(self,co_Kinv,K_inv_diag,eta_r,n_var,std_p,jac=False,multiple_para=False,r_j=None,s_j=None,qp=None,qu=False):
        "Function that checks what predective and uncertainty cost functions to use"
        if not jac:
            qp,qu=0,0
            if self.cost_args['Wp']!=0:
                qp=self.cost_pred(co_Kinv,K_inv_diag,std_p,jac=False)
            if self.cost_args['Wu']!=0:
                if self.cost_args['Qu']==1:
                    qu=self.cost_unc1(eta_r,n_var,jac=False)
                elif self.cost_args['Qu']==2:
                    qu=self.cost_unc2(eta_r,n_var,jac=False)
            return qp,qu
        qp_d,qu_d=0,0
        if self.cost_args['Wp']!=0:
            qp_d=self.cost_pred(co_Kinv,K_inv_diag,std_p,jac,qp,multiple_para,r_j,s_j)
        if self.cost_args['Wu']!=0:
            if self.cost_args['Qu']==1:
                qu_d=self.cost_unc1(eta_r,n_var,jac,qu,multiple_para,co_Kinv,r_j,s_j)
            elif self.cost_args['Qu']==2:
                qu_d=self.cost_unc2(eta_r,n_var,jac,qu,multiple_para,co_Kinv,r_j,s_j)
        return qp_d,qu_d

    def cost_combi(self,qp,qu,jac=False,cost_value=None,qp_d=None,qu_d=None):
        "Function that makes the right combination of the predective and uncertainty cost function"
        if not jac:
            if self.cost_args['combi'].lower()=='sum':
                return self.cost_args['Wp']*qp+self.cost_args['Wu']*qu
            elif self.cost_args['combi'].lower()=='exp':
                return (qp**self.cost_args['Wp'])+(qu**self.cost_args['Wu'])
            elif self.cost_args['combi'].lower()=='prod':
                return (qp**self.cost_args['Wp'])*(qu**self.cost_args['Wu'])
        if self.cost_args['combi'].lower()=='sum':
            return self.cost_args['Wp']*qp_d+self.cost_args['Wu']*qu_d
        elif self.cost_args['combi'].lower()=='exp':
            cost_dp,cost_du=0,0
            if self.cost_args['Wp']!=0:
                cost_dp=self.cost_args['Wp']*qp_d*(qp**(self.cost_args['Wp']-1))
            if self.cost_args['Wu']!=0:
                cost_du=self.cost_args['Wu']*qu_d*(qu**(self.cost_args['Wu']-1))
            return cost_dp+cost_du
        elif self.cost_args['combi'].lower()=='prod':
            cost_dp,cost_du=0,0
            if self.cost_args['Wp']!=0:
                cost_dp=self.cost_args['Wp']*qp_d/qp
            if self.cost_args['Wu']!=0:
                cost_du=self.cost_args['Wu']*qu_d/qu
            return cost_value*(cost_dp+cost_du)

    def mnll(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Minimum negative log likelihood"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        GP.set_hyperparams({'alpha':np.array([1.0])})
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        alpha_max=np.matmul(Y.T,coef).item(0)/n_data
        nmll=0.5*n_data*(1+np.log(2*np.pi)+np.log(alpha_max))+np.sum(np.log(np.diagonal(L)))
        if jac==False:
            return nmll
        # Derivatives
        nmll_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                nmll_deriv=np.append(nmll_deriv,np.array([0]*len(GP.hp['alpha'])))
                continue
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nmll_deriv=np.append(nmll_deriv,-sign_t[para]*0.5*((1/alpha_max)*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)-K_deriv_cho))
        return nmll,nmll_deriv

    def mnlp(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        "Minimum negative log posterior"
        GP,parameters_set,sign_t=self.update_gp(GP,parameters,theta)
        GP.set_hyperparams({'alpha':np.array([1.0])})
        coef,L,low,Y,KXX,n_data=self.get_coef(GP,X,Y,dis_m)
        alpha_max=np.matmul(Y.T,coef).item(0)/n_data
        nmll=0.5*n_data*(1+np.log(2*np.pi)+np.log(alpha_max))+np.sum(np.log(np.diagonal(L)))
        if prior is not None:
            nmll+=-np.sum([np.sum([pr.ln_pdf(GP.hp[para][p]) for p,pr in enumerate(prior[para])]) for para in prior.keys() if para is not 'alpha'])
            #nmll+=-2*np.sum([np.sum(prior[para].ln_pdf(GP.hp[para])) for para in sorted(list(prior.keys())) if para is not 'alpha'])/n_data
        if jac==False:
            return nmll
        # Derivatives
        nmll_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        K_derivs=GP.get_derivatives(X,parameters_set,KXX=KXX,dists=dis_m)
        for para in parameters_set:
            K_deriv=copy.deepcopy(K_derivs[para])
            if para=='alpha':
                nmll_deriv=np.append(nmll_deriv,np.array([0]*len(GP.hp['alpha'])))
                continue
            multiple_para=len(GP.hp[para])>1
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv,multiple_para)
            nmll_d=-0.5*((1/alpha_max)*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1)-K_deriv_cho)
            if prior is not None:
                nmll_d+=np.array([pr.ln_deriv(GP.hp[para][p]) for p,pr in enumerate(prior[para])])
                #nmll_d+=-2*np.array(prior[para].ln_deriv(GP.hp[para]))/n_data
            nmll_deriv=np.append(nmll_deriv,sign_t[para]*nmll_d)
        return nmll,nmll_deriv

    def update_gp(self,GP,parameters,theta):
        " Update the GP with each type of hyperparameters"
        # Each type of hyperparameters
        parameters_set=sorted(list(set(parameters)))
        # Hyperparameters
        hp={para_s:np.array([theta[p] for p,para in enumerate(parameters) if para==para_s]) for para_s in parameters_set}
        # Get the sign of the hyperparameters (used for derivatives)
        sign_t={para_s:np.array([-1 if p<0 else 1 for p in hp[para_s]]) for para_s in parameters_set}
        hp={para_s:np.abs(hp[para_s]) for para_s in parameters_set}
        # Update
        GP.set_hyperparams(hp)
        return GP,parameters_set,sign_t

    def get_coef(self,GP,X,Y,dis_m):
        " Calculate the coefficients by using Cholesky decomposition "
        # Calculate the kernel with and without noise
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dists=dis_m)
        KXX_noise=GP.add_regularization(KXX,len(X),overwrite=False)
        # Subtract the prior mean to the training target
        Y=Y.copy()
        Y[:,0]=Y[:,0]-GP.prior.get(X)
        if GP.use_derivatives:
            Y=Y.T.reshape(-1,1)
        # Cholesky decomposition
        L,low=cho_factor(KXX_noise)
        # Get the coefficients
        coef=cho_solve((L,low),Y,check_finite=False)
        return coef,L,low,Y,KXX,len(KXX)

    def get_derivative_correction(self,K_deriv,hp,sign_t,n_data):
        """ Get the contribution of the derivative of the covariance matrix 
                as respect to alpha when the noise correction is used """
        if 'correction' in hp.keys():
            if len(hp['alpha'])>1:
                for i in range(len(hp['alpha'])):
                    K_deriv[i,range(n_data),range(n_data)]+=2*hp['correction']/(sign_t['alpha'][i]*hp['alpha'][i])
            else:
                K_deriv[range(n_data),range(n_data)]+=2*hp['correction']/(sign_t['alpha'][0]*hp['alpha'])
        return K_deriv

    def get_K_inv_deriv(self,K_deriv,KXX_inv,multiple_para):
        " Get the diagonal elements of the matrix product of the inverse and derivative covariance matrix "
        if multiple_para:
            K_deriv_cho=np.array([np.einsum('ij,ji->',KXX_inv,K_d) for K_d in K_deriv])
        else:
            K_deriv_cho=np.einsum('ij,ji->',KXX_inv,K_deriv)
        return K_deriv_cho

    def get_r_s_derivatives(self,K_deriv,KXX_inv,coef,multiple_para):
        " Get the r and s vector that are products of the inverse and derivative covariance matrix "
        if multiple_para:
            r_j=np.array([-np.matmul(np.matmul(KXX_inv,K_d),coef) for K_d in K_deriv])
            s_j=np.array([np.einsum('ij,ji->i',np.matmul(KXX_inv.T,K_d),KXX_inv) for K_d in K_deriv])
        else:
            r_j=-np.matmul(np.matmul(KXX_inv,K_deriv),coef)
            s_j=np.einsum('ij,ji->i',np.matmul(KXX_inv.T,K_deriv),KXX_inv)
        return r_j,s_j

    def log_wrapper(self,theta,GP,parameters,X,Y,prior=None,jac=False,dis_m=None):
        " A wrapper that calculate the object function within the log-space of the hyperparameters "
        theta=np.exp(theta)
        values=self.func(theta,GP,parameters,X,Y,prior,jac,dis_m)
        if jac:
            parameters_set=sorted(list(set(parameters)))
            if 'correction' in parameters_set:
                parameters_set.remove('correction')
            theta=[list(np.array(GP.hp[para]).reshape(-1)) for para in parameters_set]
            parameters=sum([[para]*len(theta[p]) for p,para in enumerate(parameters_set)],[])
            theta=np.array(sum(theta,[]))
            return values[0],values[1]*theta
        return values




