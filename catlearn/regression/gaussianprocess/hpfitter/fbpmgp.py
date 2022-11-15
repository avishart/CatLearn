import numpy as np
import copy
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.optimize import OptimizeResult
from ..hptrans import Variable_Transformation

class FBPMGP:
    def __init__(self,Q=None,ngrid=100,use_bounds=True,n_test=50,distance_matrix=True):
        """ Get the best GP that mimic the Full-Bayesian predictive distribution. 
        Parameters:
            Q : (M,D) array
                Test features to check the predictive distribution 
            ngrid : int
                Number of points in each hyperparameter to evaluate the posterior distribution
            use_bounds : bool
                Use an educated bound to make grid in each hyperparameter.
            n_test : int
                n_test is used to make test features if the test features is not given
            distance_matrix : bool
                Whether to reuse the distance matrix for the optimization.
        """
        self.Q=Q
        self.ngrid=ngrid
        self.use_bounds=use_bounds
        self.n_test=n_test
        self.distance_matrix=distance_matrix
        
    def fit(self,X,Y,GP,hp=None,prior=None):
        """ Optimize the hyperparameters 
        Parameters:
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            GP : GaussianProcess
                The Gaussian Process with kernel and prior that are optimized.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            prior : dict
                A dict of prior distributions for each hyperparameter.
        """        
        if hp is None:
            hp=GP.hp.copy()
        theta,parameters=self.hp_to_theta(hp)
        gp=copy.deepcopy(GP)
        # Whether to use distance matrix
        dis_m=gp.kernel.distances(X) if self.distance_matrix else None
        sol=self.fbpmgp(theta,GP,parameters,X,Y,prior=prior,dis_m=dis_m,Q=self.Q,ngrid=self.ngrid,use_bounds=self.use_bounds)
        return sol
    
    def hp_to_theta(self,hp):
        " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
        parameters_set=sorted(set(hp.keys()))
        theta=[list(np.array(hp[para]).reshape(-1)) for para in parameters_set]
        parameters=sum([[para]*len(theta[p]) for p,para in enumerate(parameters_set)],[])
        theta=np.array(sum(theta,[]))
        return theta,parameters
        
    def hp(self,theta,parameters):
        " Make hyperparameter dictionary from lists"
        theta,parameters=np.array(theta),np.array(parameters)
        parameters_set=sorted(set(parameters))
        hp={para_s:self.numeric_limits(theta[parameters==para_s]) for para_s in parameters_set}
        return hp,parameters_set
    
    def numeric_limits(self,array,dh=0.4*np.log(np.finfo(float).max)):
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
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dists=dis_m)
        n_data=len(KXX)
        KXX=self.add_correction(GP,KXX,n_data)
        return KXX,n_data
    
    def add_correction(self,GP,KXX,n_data):
        " Add noise correction to covariance matrix"
        if GP.correction:
            corr=GP.get_correction(np.diag(KXX))
            KXX[range(n_data),range(n_data)]+=corr
        return KXX
        
    def y_prior(self,X,Y,GP):
        " Update prior and subtract target "
        Y_p=Y.copy()
        GP.prior.update(X,Y_p)
        Y_p=Y_p-GP.prior.get(X)
        if GP.use_derivatives:
            Y_p=Y_p.T.reshape(-1,1)
        return Y_p,GP
    
    def get_eig(self,GP,X,Y,dis_m):
        " Calculate the eigenvalues " 
        # Calculate the kernel with and without noise
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dists=dis_m)
        n_data=len(KXX)
        KXX[range(n_data),range(n_data)]+=GP.get_correction(np.diag(KXX))
        # Eigendecomposition
        D,U=eigh(KXX)
        # Subtract the prior mean to the training target
        Y_p,GP=self.y_prior(X,Y,GP)
        UTY=np.matmul(U.T,Y_p)
        UTY2=UTY.reshape(-1)**2
        return D,U,UTY,UTY2,Y_p,KXX,n_data,GP.prior.yp
    
    def get_eig_without_Yp(self,GP,X,Y_p,dis_m,n_data):
        " Calculate the eigenvalues without using the prior mean " 
        # Calculate the kernel with and without noise
        KXX=GP.kernel(X,get_derivatives=GP.use_derivatives,dists=dis_m)
        KXX[range(n_data),range(n_data)]+=GP.get_correction(np.diag(KXX))
        # Eigendecomposition
        D,U=eigh(KXX)
        UTY=np.matmul(U.T,Y_p)
        UTY2=UTY.reshape(-1)**2
        return D,U,UTY,UTY2,Y_p,KXX
    
    def pred_unc(self,UKQX,UTY,D_n,KQQ,yp):
        " Make prediction mean and uncertainty from eigendecomposition "
        UKQXD=UKQX/D_n
        pred=yp+np.matmul(UKQXD,UTY)
        var=(KQQ-np.einsum('ij,ji->i',UKQXD,UKQX.T))
        return pred.reshape(-1),var.reshape(-1)
    
    def get_grids(self,GP,X,Y,parameters_set,para_bool,ngrid=100,use_bounds=True):
        " Make a grid for each hyperparameter in the variable transformed space "
        hyper_var=Variable_Transformation().transf_para(parameters_set,GP,X,Y,use_bounds=use_bounds)
        dl=np.finfo(float).eps
        grids={}
        for para,pbool in para_bool.items():
            if pbool:
                grids[para]=hyper_var.transform_t_to_hyper(np.linspace(0.0+dl,1.0-dl,ngrid),para)
            else:
                grids[para]=np.array([GP.hp['prefactor'].item(0)])
        return grids
    
    def trapz_coef(self,grids,para_bool):
        " Make the weights for the weighted averages from the trapezoidal rule "
        cs={}
        for para,pbool in para_bool.items():
            if pbool:
                cs[para]=np.log(self.trapz_append(grids[para]))
            else:
                cs[para]=np.array([0.0])
        return cs
    
    def prior_grid(self,grids,prior=None,i=0):
        " Get prior distribution of hyperparameters on the grid "
        if prior is None:
            return {para:np.array([0.0]*len(grid)) for para,grid in grids.items()}
        pr_grid={}
        for para,grid in grids.items():
            if para in prior.keys():
                pr_grid[para]=prior[para][i].ln_pdf(grid)
            else:
                pr_grid[para]=np.array([0.0]*len(grid))
        return pr_grid
    
    def get_all_grids(self,parameters_set,GP,X,Y_p,ngrid=100,use_bounds=True,prior=None):
        " Get the grids in the hyperparameter space, weights from the trapezoidal rule, and prior grid "
        # Check whether all hyperparameters are optimized or fixed
        parameters_need=['length','noise','prefactor']
        para_bool={para:para in parameters_set for para in parameters_need}
        # Make grid and transform hyperparameters into another space
        grids=self.get_grids(GP,X,Y_p,parameters_set,para_bool,ngrid=ngrid,use_bounds=use_bounds)
        # Make the weights for the weighted averages 
        cs=self.grid_sum_pn(self.trapz_coef(grids,para_bool))
        pr_grid=self.grid_sum_pn(self.prior_grid(grids,prior))
        return grids,cs,pr_grid
    
    def trapz_append(self,grid):
        " Get the weights in linear space from the trapezoidal rule "
        return np.append([grid[1]-grid[0]],np.append(grid[2:]-grid[:-2],grid[-1]-grid[-2]))*0.5
    
    def get_test_points(self,Q,X_tr):
        " Get the test point if they are not given "
        if Q is not None:
            return Q
        i_sort=np.argsort(pdist(X_tr))[:self.n_test]
        i_list,j_list=np.triu_indices(len(X_tr),k=1,m=None)
        i_list,j_list=i_list[i_sort],j_list[i_sort]
        r=np.random.uniform(low=0.01,high=0.99,size=(2,len(i_list)))
        r=r/np.sum(r,axis=0)
        return np.array([X_tr[i]*r[0,k]+X_tr[j]*r[1,k] for k,(i,j) in enumerate(zip(i_list,j_list))])
    
    def get_test_KQ(self,GP,Q,X_tr,use_derivatives=False):
        " Get the test point if they are not given and get the covariance matrix "
        Q=self.get_test_points(Q,X_tr).copy()
        KQQ=GP.kernel.diag(Q,get_derivatives=use_derivatives)
        return Q,KQQ
    
    def get_prefactors(self,grids,n_data):
        " Get the prefactor values for log-likelihood "
        prefactors=np.exp(2*grids['prefactor']).reshape(-1,1)
        ln_prefactor=(n_data*grids['prefactor']).reshape(-1,1)
        return prefactors,ln_prefactor
    
    def grid_sum_pn(self,the_grids):
        " Make a grid of prefactor and noise at the same time and a grid of length-scale "
        return {'length':the_grids['length'],'np':the_grids['prefactor'].reshape(-1,1)+the_grids['noise']}
    
    def get_grid_sum(self,the_grids,l,n):
        " Sum together the grid value of length-scale and the merged prefactor and noise grid "
        return the_grids['length'][l]+the_grids['np'][:,n:n+1]
    
    def get_all_eig_matrices(self,length,GP,X,Y_p,dis_m,n_data,Q,get_derivatives=False):
        " Get all the matrices from eigendecomposition that must be used to posterior distribution and predictions "
        GP.set_hyperparams({'length':[length]})
        D,U,UTY,UTY2,Y_p,KXX=self.get_eig_without_Yp(GP,X,Y_p,dis_m,n_data)
        KQX=GP.kernel(Q,X,get_derivatives=get_derivatives)
        UKQX=np.matmul(KQX,U)
        return D,UTY,UTY2,UKQX
    
    def posterior_value(self,like_sum,lp_max,UTY2,D_n,prefactors,ln_prefactor,ln2pi,pr_grid,cs,l,n):
        " Get the posterior distribution value and add it to the existing sum "
        nlp1=0.5*np.sum(UTY2/D_n)
        nlp2=0.5*np.sum(np.log(D_n))
        like=-((nlp1/prefactors+ln_prefactor)+(nlp2+ln2pi))+self.get_grid_sum(pr_grid,l,n)
        like_max=np.nanmax(like)
        if like_max>lp_max:
            ll_scale=np.exp(lp_max-like_max)
            lp_max=like_max
        else:
            ll_scale=1.0
        like=like-lp_max
        like=np.exp(like+self.get_grid_sum(cs,l,n))
        like_sum=like_sum*ll_scale+np.sum(like)
        return like_sum,like,lp_max,ll_scale
    
    def update_df_ybar(self,df,ybar,y2bar_ubar,pred,var,like,ll_scale,prefactors,length,noise):
        " Update the dict and add values to ybar and y2bar_ubar "
        ybar=(ybar*ll_scale)+np.sum(pred*like,axis=0)
        y2bar_ubar=(y2bar_ubar*ll_scale)+(np.sum((pred**2)*like,axis=0)+np.sum(prefactors*var*like,axis=0))        
        # Store the hyperparameters and prediction mean and variance
        df['length'].append(length)
        df['noise'].append(noise)
        df['pred'].append(pred)
        df['var'].append(var)
        return df,ybar,y2bar_ubar
    
    def get_solution(self,df,ybar,y2bar_ubar,like_sum,n_test,GP,len_l):
        " Find the hyperparameters that gives the lowest Kullback-Leibler divergence "
        # Make df to numpy array
        df['pred']=np.array(df['pred'])
        df['var']=np.array(df['var'])
        # Normalize the weighted sums
        ybar=ybar/like_sum
        y2bar_ubar=y2bar_ubar/like_sum
        # Get the analytic solution to the prefactor 
        prefactor=np.mean((y2bar_ubar+(df['pred']**2)-(2*df['pred']*ybar))/df['var'],axis=1)
        # Calculate all Kullback-Leibler divergences
        kl=0.5*(n_test*(1+np.log(2*np.pi))+(np.sum(np.log(df['var']),axis=1)+n_test*np.log(prefactor)))
        # Find the best solution
        i_min=np.nanargmin(kl)
        kl_min=kl[i_min]/n_test
        hp_best=dict(length=np.array([df['length'][i_min]]),noise=np.array([df['noise'][i_min]]),prefactor=np.array([0.5*np.log(prefactor[i_min])]))
        theta=np.array([hp_best[para] for para in hp_best.keys()]).reshape(-1)
        sol={'fun':kl_min,'hp':hp_best,'x':theta,'nfev':len_l,'success':True}
        sol['GP']=self.update(GP,hp_best)
        return sol
    
    def fbpmgp(self,theta,GP,parameters,X,Y,prior=None,dis_m=None,Q=None,ngrid=100,use_bounds=True):
        " Only works with the FBPMGP object function " 
        # Update hyperparameters
        hp,parameters_set=self.hp(theta,parameters)
        GP=self.update(GP,hp)
        # Update prior mean 
        Y_p,GP=self.y_prior(X,Y,GP)
        yp=GP.prior.yp
        use_derivatives=GP.use_derivatives
        # Make grids of hyperparameters, weights from the trapezoidal rule, and prior distribution grid
        grids,cs,pr_grid=self.get_all_grids(parameters_set,GP,X,Y_p,ngrid=ngrid,use_bounds=use_bounds,prior=prior)
        # Use test data
        Q,KQQ=self.get_test_KQ(GP,Q,X,use_derivatives=use_derivatives)
        n_data,n_test=len(Y_p),len(KQQ)
        # 
        df={'ll':[],'length':[],'noise':[],'prefactor':[],'pred':[],'var':[]}
        like_sum,ybar,y2bar_ubar=0,0,0
        lp_max=-np.inf
        prefactors,ln_prefactor=self.get_prefactors(grids,n_data)
        ln2pi=0.5*n_data*np.log(2*np.pi)
        for l,length in enumerate(grids['length']):
            D,UTY,UTY2,UKQX=self.get_all_eig_matrices(length,GP,X,Y_p,dis_m,n_data,Q,get_derivatives=use_derivatives)
            for n,noise in enumerate(grids['noise']):
                D_n=D+np.exp(2*noise)
                # Calculate log-posterior
                like_sum,like,lp_max,ll_scale=self.posterior_value(like_sum,lp_max,UTY2,D_n,prefactors,ln_prefactor,ln2pi,pr_grid,cs,l,n)
                # Calculate prediction mean and variance
                pred,var=self.pred_unc(UKQX,UTY,D_n,KQQ,yp)
                # Store and update the hyperparameters and prediction mean and variance
                df,ybar,y2bar_ubar=self.update_df_ybar(df,ybar,y2bar_ubar,pred,var,like,ll_scale,prefactors,length,noise)
        sol=self.get_solution(df,ybar,y2bar_ubar,like_sum,n_test,GP,len(grids['length']))
        return OptimizeResult(**sol)

                
                
