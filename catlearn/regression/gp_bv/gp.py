import numpy as np
import copy
from scipy.linalg import cho_factor,cho_solve

class GaussianProcess:
    def __init__(self,prior=None,kernel=None,hp={'noise':1e-4},use_derivatives=False,hpfitter=None):
        """The Gaussian Process Regression solver with Cholesky decomposition and optimization of hyperparameters.
            Parameters:
                prior : Prior class
                    The prior given for new data.
                kernel : Kernel class
                    The kernel function used for the kernel matrix.
                hp : dictionary
                    A dictionary of hyperparameters like noise, length scale, and prefactor.
                use_derivatives : bool
                    Use derivatives/gradients for training and predictions
        """
        #Kernel
        if kernel is None:
            from catlearn.regression.gp_bv.kernel import SE,SE_Deriv
            self.kernel=copy.deepcopy(SE_Deriv()) if use_derivatives else copy.deepcopy(SE())
        else:
            self.kernel=copy.deepcopy(kernel)
        #Prior
        if prior is None:
            from catlearn.regression.gp_bv.prior import Prior_mean
            self.prior=copy.deepcopy(Prior_mean())
        else:
            self.prior=copy.deepcopy(prior)
        #Whether to use derivatives or not for the target
        self.use_derivatives=use_derivatives
        #The hyperparameter optimization method
        if hpfitter is None:
            from catlearn.regression.gp_bv.optimization import HyperparameterFitter
            hpfitter=HyperparameterFitter()
        self.hpfitter=hpfitter    
        #Set hyperparameters
        self.hp={}
        self.set_hyperparams(hp)

    
    def train(self,X,Y):
        """Train the Gaussian process with training features and targets. 
        Parameters:
            X : (N,D) array
                Training features with N data points and D dimensions
            Y : (N,1) array
                Training targets with N data points 
            or 
            Y : (N,1+D) array
                Training targets in first column and derivatives of each feature in the next columns if use_derivatives is True
        Returns trained Gaussian process:
        """
        Y=Y.copy()
        #Prior mean on targets
        self.prior.update(X,Y[:,0])
        Y[:,0]=Y[:,0]-self.prior.get(X)
        #Rearrange targets if derivatives are used 
        if self.use_derivatives:
            Y=Y.T.reshape(-1,1)
        self.X=X.copy()
        #Make kernel matrix
        K=self.kernel(X,get_derivatives=True)
        #Add noise to the diagonal
        K=self.add_regularization(K,len(X))
        #Calculate the coefficients
        self.L,self.low=cho_factor(K)
        self.coef=cho_solve((self.L,self.low),Y,check_finite=False)
        return self

    def predict(self,Q,get_variance=False,get_derivatives=False):
        """Predict the mean and variance for test features by using data and coefficients from training data.
        Parameters:
            Q : (M,D) array
                Test features with M data points and D dimensions
            get_variance : bool
                Whether to predict the vartiance
            get_derivatives : bool
                Whether to predict the derivative mean and uncertainty
        Returns:
            Y_predict : (M,1) array 
                The predicted mean values
            or 
            Y_predict : (M,1+D) array
                The predicted mean values and derivatives
            var : (M,1) array
                The predicted variance of values
            or 
            var : (M,1+D) array
                The predicted variance of values and derivatives
        """
        #Calculate the kernel matrix of test and training data
        KQX=self.kernel(Q,self.X,get_derivatives=get_derivatives)
        n_data=len(Q)
        #Calculate the predicted values
        Y_predict=np.matmul(KQX,self.coef)
        Y_predict[:n_data]+=self.prior.get(Q)
        #Check if the derivatives are calculated
        if len(Y_predict)==n_data:
            get_derivatives=False
        Y_predict=Y_predict.reshape(n_data,-1,order='F')
        #Calculate the predicted variance
        if get_variance:
            var=self.calculate_variance(Q,KQX,get_derivatives=get_derivatives)
            return Y_predict,var
        return Y_predict

    def calculate_variance(self,Q,KQX,get_derivatives=False):
        """Calculate the predicted variance
        Parameters:
            Q : (M,D) array
                Test features with M data points and D dimensions.
            KQX : (M,N) array or (M*(1+D),N*(1+D)) array or (M,N*(1+D))
                The kernel matrix of test and training data.
            get_derivatives : bool
                Whether to predict the derivative uncertainty.
        Returns:
            var : (M,1) array
                The predicted variance of values.
            or 
            var : (M,1+D) array
                The predicted variance of values and derivatives.
        """
        #Calculate the diagonal elements of the kernel matrix with noise and correction of the test points
        n_data=len(Q)
        k=self.kernel.diag(Q,get_derivatives=get_derivatives)+self.hp['correction']
        if get_derivatives and self.use_derivatives and 'noise_deriv' in self.hp:
            k[:n_data]+=self.hp['noise']**2
            k[n_data:]+=self.hp['noise_deriv']**2
        else:
            k+=self.hp['noise']**2
        #Calculate predicted variance
        var=(k-np.einsum('ij,ji->i',KQX,cho_solve((self.L,self.low),KQX.T,check_finite=False))).reshape(-1,1)
        if get_derivatives and self.use_derivatives:
            return var.reshape(n_data,-1,order='F')
        return var

    def optimize(self,X,Y,retrain=True,hp=None,maxiter=None,prior=None,verbose=False):
        """ Optimize the hyperparameter of the Gaussian Process and its kernel
        Parameters:
            X : (N,D) array
                Training features with N data points and D dimensions.
            Y : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            retrain : bool
                Whether to retrain the Gaussian Process after the optimization.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            maxiter : int
                Maximum number of iterations used by local or global optimization method.
            prior : dict
                A dict of prior distributions for each hyperparameter
            verbose : bool
                Print the optimized hyperparameters and the object function value
        """
        GP=copy.deepcopy(self)
        sol=self.hpfitter.fit(X,Y,GP,hp=hp,maxiter=maxiter,prior=prior)
        if verbose:
            print(sol)
        if retrain:
            self.set_hyperparams(sol['hp'])
            self.train(X,Y)
        return sol


    def add_regularization(self,K,n_data,overwrite=True):
        "Add the regularization to the diagonal elements of the squared kernel matrix. (K will be overwritten if overwrite=True)"
        #Calculate the correction, so the kernel matrix is invertible
        if not overwrite:
            K=K.copy()
        self.hp['correction']=np.array([self.get_correction(np.diag(K))])
        m_data=len(K)
        if self.use_derivatives and 'noise_deriv' in self.hp:
            K[range(n_data),range(n_data)]+=self.hp['noise']**2+self.hp['correction']
            K[range(n_data,m_data),range(n_data,m_data)]+=self.hp['noise_deriv']**2+self.hp['correction']
        else:
            K[range(m_data),range(m_data)]+=self.hp['noise']**2+self.hp['correction']
        return K


    def set_hyperparams(self,new_params):
        """Set or update the hyperparameters for the GP.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
            Returns:
                hp : dictionary
                    An updated dictionary of hyperparameters with noise, kernel hyperparameters (like length, and alpha) 
                    and noise_deriv for the derivative part of the kernel if specified.
        """
        self.hp.update(new_params)
        self.hp=self.kernel.set_hyperparams(self.hp)
        #Upper machine precision
        eps_mach_upper=1/np.sqrt(1.01*np.finfo(float).eps)
        #Noise is always in the GP, but must be lower that upper machine precision
        if 'noise' not in self.hp:
            self.hp['noise']=np.array([1e-4])
        self.hp['noise']=np.abs(self.hp['noise']).reshape(-1)
        self.hp['noise']=np.where(self.hp['noise']<eps_mach_upper,self.hp['noise'],eps_mach_upper)
        #If noise_deriv is used to have a specific noise for the derivative part only 
        if self.use_derivatives and 'noise_deriv' in self.hp:
            self.hp['noise_deriv']=np.abs(self.hp['noise_deriv']).reshape(-1)
            self.hp['noise_deriv']=np.where(self.hp['noise_deriv']<eps_mach_upper,self.hp['noise_deriv'],eps_mach_upper)
        return self.hp
            
            
    def get_correction(self,K_diag):
        "Get the correction, so that the training covariance matrix is always invertible"
        return np.sum(K_diag)*(1/(1/(4.0*np.finfo(float).eps)-1))


    def get_hyperparameters(self):
        "Get all the hyperparameters"
        return self.hp


    def get_derivatives(self,X,hp,KXX=None,dists=None):
        """Get the derivatives of the Gaussian Process in respect to the hyperparameters.
            Parameters:
                X : (N,D) array
                    Features with N data points and D dimensions.
                hp : list
                    A list with elements of the hyperparameters that are optimized.
                KXX : (N,N) array (optional)
                    The kernel matrix of training data.
                dists : (N,N) array (optional)
                    Can be given the distance matrix to avoid recaulcating it.
        """
        hp_deriv={}
        if 'noise' in hp and 'noise_deriv' not in self.hp:
            if KXX is None:
                KXX=self.kernel(X)
            hp_deriv['noise']=np.diag(np.array([2*self.hp['noise']]*len(KXX)).reshape(-1))
        if 'noise' in hp and 'noise_deriv' in self.hp:
            if KXX is None:
                KXX=self.kernel(X)
            hp_deriv['noise']=np.diag(np.array([2*self.hp['noise']]*len(X)+[0]*int(len(KXX)-len(X))).reshape(-1))
        if 'noise_deriv' in hp:
            if KXX is None:
                KXX=self.kernel(X)
            hp_deriv['noise_deriv']=np.diag(np.array([0]*len(X)+[2*self.hp['noise_deriv']]*int(len(KXX)-len(X))).reshape(-1))
        hp_deriv.update(self.kernel.get_derivatives(X,hp,KXX=KXX,dists=dists))
        return hp_deriv


    def __repr__(self):
        return "GP({} ; use_derivatives={})".format(self.hp,self.use_derivatives)

            
        








