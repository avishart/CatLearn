import numpy as np
from numpy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.optimize import OptimizeResult
from .hpfitter import HyperparameterFitter

class FBPMGP(HyperparameterFitter):
    def __init__(self,Q=None,n_test=50,ngrid=80,bounds=None,get_prior_mean=False,**kwargs):
        """ 
        Get the best Gaussian Process that mimic the Full-Bayesian predictive distribution. 
        It only works with a Gaussian Process.
        Parameters:
            Q: (M,D) array
                Test features to check the predictive distribution 
            n_test: int (optional)
                n_test is used to make test features if the test features is not given
            ngrid: int
                Number of points in each hyperparameter to evaluate the posterior distribution
            bounds: Boundary_conditions class
                A class that calculates the boundary conditions of the hyperparameter.
            get_prior_mean: bool
                Whether to get the prior arguments in the solution.
        """
        # Set the default test points
        self.Q=None
        # Set the default boundary conditions
        if bounds is None:
            from ..hpboundary.hptrans import VariableTransformation
            self.bounds=VariableTransformation(bounds=None)
        # Set the solution form
        self.update_arguments(Q=Q,
                              n_test=n_test,
                              ngrid=ngrid,
                              bounds=bounds,
                              get_prior_mean=get_prior_mean,
                              **kwargs)
        
    def fit(self,X,Y,model,hp=None,pdis=None,**kwargs):
        """ 
        Optimize the hyperparameters 
        Parameters:
            X: (N,D) array
                Training features with N data points and D dimensions.
            Y: (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            model: Model Process
                The Gaussian Process with kernel and prior that are optimized.
            hp: dict
                Use a set of hyperparameters to optimize from else the current set is used.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
        Returns: 
            dict: A solution dictionary with objective function value, optimized hyperparameters,
                success statement, and number of used evaluations.
        """
        # Copy the model so it is not changed outside of the optimization
        model=self.copy_model(model)
        # Get hyperparameters
        hp,theta,parameters=self.get_hyperparams(hp,model)
        # Find FBMGP solution
        sol=self.fbpmgp(theta,model,parameters,X,Y,pdis=pdis,Q=self.Q,ngrid=self.ngrid)
        sol=self.get_full_hp(sol,model)
        return sol
    
    def update_arguments(self,Q=None,n_test=None,ngrid=None,bounds=None,get_prior_mean=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            Q: (M,D) array
                Test features to check the predictive distribution 
            n_test: int (optional)
                n_test is used to make test features if the test features is not given
            ngrid: int
                Number of points in each hyperparameter to evaluate the posterior distribution
            bounds: Boundary_conditions class
                A class that calculates the boundary conditions of the hyperparameter.
            get_prior_mean: bool
                Whether to get the prior arguments in the solution.
        Returns:
            self: The updated object itself.
        """
        if Q is not None:
            self.Q=Q.copy()
        if n_test is not None:
            self.n_test=int(n_test)
        if ngrid is not None:
            self.ngrid=int(ngrid)
        if bounds is not None:
            self.bounds=bounds.copy()
        if get_prior_mean is not None:
            self.get_prior_mean=get_prior_mean
        return self
        
    def get_hp(self,theta,parameters,**kwargs):
        " Make hyperparameter dictionary from lists"
        theta,parameters=np.array(theta),np.array(parameters)
        parameters_set=sorted(set(parameters))
        hp={para_s:self.numeric_limits(theta[parameters==para_s]) for para_s in parameters_set}
        return hp,parameters_set
    
    def numeric_limits(self,array,dh=0.4*np.log(np.finfo(float).max)):
        " Replace hyperparameters if they are outside of the numeric limits in log-space "
        return np.where(-dh<array,np.where(array<dh,array,dh),-dh)
    
    def update_model(self,model,hp,**kwargs):
        " Update model "
        model.set_hyperparams(hp)
        return model
    
    def kxx_corr(self,model,X,**kwargs):
        " Get covariance matrix with or without noise correction"
        # Calculate the kernel with and without noise
        KXX=model.get_kernel(X,get_derivatives=model.use_derivatives)
        n_data=len(KXX)
        KXX=self.add_correction(model,KXX,n_data)
        return KXX,n_data
    
    def add_correction(self,model,KXX,n_data,**kwargs):
        " Add noise correction to covariance matrix"
        corr=model.get_correction(np.diag(KXX))
        if corr!=0.0:
            KXX[range(n_data),range(n_data)]+=corr
        return KXX
        
    def y_prior(self,X,Y,model,L=None,low=None,**kwargs):
        " Update prior and subtract target "
        Y_p=Y.copy()
        model.update_priormean(X,Y_p,L=L,low=low,**kwargs)
        if model.use_derivatives:
            return (Y_p-model.get_priormean(X,Y_p,get_derivatives=True)).T.reshape(-1,1)
        return (Y_p-model.get_priormean(X,Y_p,get_derivatives=False))[:,0:1]
    
    def get_eig(self,model,X,Y,**kwargs):
        " Calculate the eigenvalues. " 
        # Calculate the kernel with and without noise
        KXX,n_data=self.kxx_corr(model,X)
        # Eigendecomposition
        try:
            D,U=eigh(KXX)
        except Exception as e:
            import logging
            import scipy.linalg
            logging.error("An error occurred: %s", str(e))
            # More robust but slower eigendecomposition
            D,U=scipy.linalg.eigh(KXX,driver='ev')
        # Subtract the prior mean to the training target
        Y_p=self.y_prior(X,Y,model,D=D,U=U)
        UTY=(np.matmul(U.T,Y_p)).reshape(-1)**2
        return D,U,Y_p,UTY,KXX,n_data
    
    def get_eig_without_Yp(self,model,X,Y_p,n_data,**kwargs):
        " Calculate the eigenvalues without using the prior mean " 
        # Calculate the kernel with and without noise
        KXX,n_data=self.kxx_corr(model,X)
        # Eigendecomposition
        try:
            D,U=eigh(KXX)
        except Exception as e:
            import logging
            import scipy.linalg
            logging.error("An error occurred: %s", str(e))
            # More robust but slower eigendecomposition
            D,U=scipy.linalg.eigh(KXX,driver='ev')
        UTY=np.matmul(U.T,Y_p)
        UTY2=UTY.reshape(-1)**2
        return D,U,UTY,UTY2,Y_p,KXX
    
    def get_grids(self,model,X,Y,parameters,para_bool,ngrid=100,**kwargs):
        " Make a grid for each hyperparameter in the variable transformed space "
        self.bounds.update_bounds(model,X,Y,parameters)
        lines=self.bounds.make_lines(ngrid=ngrid)
        grids={}
        model_hp=model.get_hyperparams()
        for p,para in enumerate(parameters):
            if para_bool[para]:
                grids[para]=lines[p].copy()
            else:
                grids[para]=np.array([model_hp[para][0]])
        return grids
    
    def trapz_coef(self,grids,para_bool,**kwargs):
        " Make the weights for the weighted averages from the trapezoidal rule "
        cs={}
        for para,pbool in para_bool.items():
            if pbool:
                cs[para]=np.log(self.trapz_append(grids[para]))
            else:
                cs[para]=np.array([0.0])
        return cs
    
    def prior_grid(self,grids,pdis=None,i=0,**kwargs):
        " Get prior distribution of hyperparameters on the grid "
        if pdis is None:
            return {para:np.array([0.0]*len(grid)) for para,grid in grids.items()}
        pr_grid={}
        for para,grid in grids.items():
            if para in pdis.keys():
                pr_grid[para]=pdis[para].ln_pdf(grid)
            else:
                pr_grid[para]=np.array([0.0]*len(grid))
        return pr_grid
    
    def get_all_grids(self,parameters_set,model,X,Y,ngrid=100,pdis=None,**kwargs):
        " Get the grids in the hyperparameter space, weights from the trapezoidal rule, and prior grid "
        # Check whether all hyperparameters are optimized or fixed
        parameters_need=sorted(['length','noise','prefactor'])
        para_bool={para:para in parameters_set for para in parameters_need}
        # Make grid and transform hyperparameters into another space
        grids=self.get_grids(model,X,Y,parameters_need,para_bool,ngrid=ngrid)
        # Make the weights for the weighted averages 
        cs=self.grid_sum_pn(self.trapz_coef(grids,para_bool))
        pr_grid=self.grid_sum_pn(self.prior_grid(grids,pdis))
        return grids,cs,pr_grid
    
    def trapz_append(self,grid,**kwargs):
        " Get the weights in linear space from the trapezoidal rule "
        return np.append([grid[1]-grid[0]],np.append(grid[2:]-grid[:-2],grid[-1]-grid[-2]))*0.5
    
    def get_test_points(self,Q,X_tr,**kwargs):
        " Get the test point if they are not given "
        if Q is not None:
            return Q
        i_sort=np.argsort(pdist(X_tr))[:self.n_test]
        i_list,j_list=np.triu_indices(len(X_tr),k=1,m=None)
        i_list,j_list=i_list[i_sort],j_list[i_sort]
        r=np.random.uniform(low=0.01,high=0.99,size=(2,len(i_list)))
        r=r/np.sum(r,axis=0)
        return np.array([X_tr[i]*r[0,k]+X_tr[j]*r[1,k] for k,(i,j) in enumerate(zip(i_list,j_list))])
    
    def get_test_KQ(self,model,Q,X_tr,use_derivatives=False,**kwargs):
        " Get the test point if they are not given and get the covariance matrix "
        Q=self.get_test_points(Q,X_tr).copy()
        KQQ=model.kernel_diag(Q,len(Q),get_derivatives=use_derivatives,include_noise=False)
        return Q,KQQ
    
    def get_prefactors(self,grids,n_data,**kwargs):
        " Get the prefactor values for log-likelihood "
        prefactors=np.exp(2*grids['prefactor']).reshape(-1,1)
        ln_prefactor=(n_data*grids['prefactor']).reshape(-1,1)
        return prefactors,ln_prefactor
    
    def grid_sum_pn(self,the_grids,**kwargs):
        " Make a grid of prefactor and noise at the same time and a grid of length-scale "
        return {'length':the_grids['length'],'np':the_grids['prefactor'].reshape(-1,1)+the_grids['noise']}
    
    def get_all_eig_matrices(self,length,model,X,Y_p,n_data,Q,get_derivatives=False,**kwargs):
        " Get all the matrices from eigendecomposition that must be used to posterior distribution and predictions "
        model.set_hyperparams({'length':[length]})
        # Training part
        D,U,UTY,UTY2,Y_p,KXX=self.get_eig_without_Yp(model,X,Y_p,n_data)
        # Test part
        KQQ=model.kernel_diag(Q,len(Q),get_derivatives=get_derivatives,include_noise=False)
        KQX=model.get_kernel(Q,X,get_derivatives=get_derivatives)
        UKQX=np.matmul(KQX,U)
        return D,UTY,UTY2,KQQ,UKQX
    
    def posterior_value(self,like_sum,lp_max,UTY2,D_n,prefactors,ln_prefactor,ln2pi,pr_grid,cs,l,**kwargs):
        " Get the posterior distribution value and add it to the existing sum "
        nlp1=0.5*np.sum(UTY2/D_n,axis=1)
        nlp2=0.5*np.sum(np.log(D_n),axis=1)
        like=-((nlp1/prefactors+ln_prefactor)+(nlp2+ln2pi))+self.get_grid_sum(pr_grid,l)
        like_max=np.nanmax(like)
        if like_max>lp_max:
            ll_scale=np.exp(lp_max-like_max)
            lp_max=like_max
        else:
            ll_scale=1.0
        like=like-lp_max
        like=np.exp(like+self.get_grid_sum(cs,l))
        like_sum=like_sum*ll_scale+np.sum(like)
        return like_sum,like,lp_max,ll_scale
    
    def get_grid_sum(self,the_grids,l):
        " Sum together the grid value of length-scale and the merged prefactor and noise grid "
        return the_grids['length'][l]+the_grids['np']
    
    def pred_unc(self,UKQX,UTY,D_n,KQQ,yp,**kwargs):
        " Make prediction mean and uncertainty from eigendecomposition "
        UKQXD=UKQX/D_n[:,None,:]
        pred=yp+np.einsum('dij,ji->di',UKQXD,UTY,optimize=True)
        var=(KQQ-np.einsum('dij,ji->di',UKQXD,UKQX.T))
        return pred,var
    
    def update_df_ybar(self,df,ybar,y2bar_ubar,pred,var,like,ll_scale,prefactors,length,noises,**kwargs):
        " Update the dict and add values to ybar and y2bar_ubar "
        ybar=(ybar*ll_scale)+np.einsum('nj,pn->j',pred,like)
        y2bar_ubar=(y2bar_ubar*ll_scale)+(np.einsum('nj,pn->j',pred**2,like)+np.einsum('nj,pn->j',var,prefactors*like))     
        # Store the hyperparameters and prediction mean and variance
        df['length']=np.append(df['length'],np.full(np.shape(noises),length))
        df['noise']=np.append(df['noise'],noises)
        df['pred']=np.append(df['pred'],pred,axis=0)
        df['var']=np.append(df['var'],var,axis=0)
        return df,ybar,y2bar_ubar
    
    def evaluate_for_noise(self,df,ybar,y2bar_ubar,like_sum,lp_max,grids,UTY,UTY2,D,UKQX,KQQ,yp,prefactors,ln_prefactor,ln2pi,pr_grid,cs,l,length,**kwargs):
        " Evaluate log-posterior and update the data frame for all noise hyperparameter in grid simulatenously. "
        D_n=D+np.exp(2*grids['noise']).reshape(-1,1)
        # Calculate log-posterior
        like_sum,like,lp_max,ll_scale=self.posterior_value(like_sum,lp_max,UTY2,D_n,prefactors,ln_prefactor,ln2pi,pr_grid,cs,l)
        # Calculate prediction mean and variance
        pred,var=self.pred_unc(UKQX,UTY,D_n,KQQ,yp)
        # Store and update the hyperparameters and prediction mean and variance
        df,ybar,y2bar_ubar=self.update_df_ybar(df,ybar,y2bar_ubar,pred,var,like,ll_scale,prefactors,length,grids['noise'])
        return df,ybar,y2bar_ubar,like_sum,lp_max
    
    def get_solution(self,df,ybar,y2bar_ubar,like_sum,n_test,model,len_l,**kwargs):
        " Find the hyperparameters that gives the lowest Kullback-Leibler divergence "
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
        if self.get_prior_mean:
            sol['prior']=model.get_prior_parameters()
        return sol
    
    def fbpmgp(self,theta,model,parameters,X,Y,pdis=None,Q=None,ngrid=100,**kwargs):
        " Only works with the FBPMGP object function " 
        np.random.seed(12)
        # Update hyperparameters
        hp,parameters_set=self.get_hp(theta,parameters)
        model=self.update_model(model,hp)
        # Make grids of hyperparameters, weights from the trapezoidal rule, and prior distribution grid
        grids,cs,pr_grid=self.get_all_grids(parameters_set,model,X,Y,ngrid=ngrid,pdis=pdis)
        # Get test data
        Q=self.get_test_points(Q,X).copy()
        # Update prior mean 
        Y_p=self.y_prior(X,Y,model)
        use_derivatives=model.use_derivatives
        yp=model.get_priormean(Q,np.zeros((len(Q),len(Y[0]))),get_derivatives=use_derivatives).reshape(-1)
        n_data=len(Y_p)
        # Initialize fb
        df={key:np.array([]) for key in ['ll','length','noise','prefactor']}
        if model.use_derivatives:
            df['pred']=np.empty((0,len(Q)*len(Y[0])))
            df['var']=np.empty((0,len(Q)*len(Y[0])))
        else:
            df['pred']=np.empty((0,len(Q)))
            df['var']=np.empty((0,len(Q)))
        like_sum,ybar,y2bar_ubar=0.0,0.0,0.0
        lp_max=-np.inf
        prefactors,ln_prefactor=self.get_prefactors(grids,n_data)
        ln2pi=0.5*n_data*np.log(2*np.pi)
        for l,length in enumerate(grids['length']):
            D,UTY,UTY2,KQQ,UKQX=self.get_all_eig_matrices(length,model,X,Y_p,n_data,Q,get_derivatives=use_derivatives)
            df,ybar,y2bar_ubar,like_sum,lp_max=self.evaluate_for_noise(df,ybar,y2bar_ubar,like_sum,lp_max,grids,UTY,UTY2,D,UKQX,KQQ,yp,prefactors,ln_prefactor,ln2pi,pr_grid,cs,l,length)
        sol=self.get_solution(df,ybar,y2bar_ubar,like_sum,len(KQQ),model,len(grids['length']))
        return OptimizeResult(**sol)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(Q=self.Q,
                        n_test=self.n_test,
                        ngrid=self.ngrid,
                        bounds=self.bounds,
                        get_prior_mean=self.get_prior_mean)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    