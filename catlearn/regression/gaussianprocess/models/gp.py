import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .model import ModelProcess

class GaussianProcess(ModelProcess):
    def __init__(self,prior=None,kernel=None,hp={},use_derivatives=False,correction=True,hpfitter=None,**kwargs):
        """The Gaussian Process Regressor solver with Cholesky decomposition and optimization of hyperparameters.
            Parameters:
                prior: Prior class
                    The prior given for new data.
                kernel: Kernel class
                    The kernel function used for the kernel matrix.
                hp: dictionary
                    A dictionary of hyperparameters like noise, length scale, and prefactor.
                use_derivatives: bool
                    Use derivatives/gradients for training and predictions.
                hpfitter: HyperparameterFitter class
                    A class to optimize hyperparameters
        """
        # Set the prior mean class
        if prior is None:
            from ..means.mean import Prior_mean
            prior=Prior_mean()
        self.prior=prior.copy()
        # Set the kernel class
        if kernel is None:
            from ..kernel import SE
            kernel=SE(use_derivatives=use_derivatives,use_fingerprint=False)
        self.kernel=kernel.copy()
        #Whether to use derivatives or not for the target
        self.use_derivatives=use_derivatives
        # Use noise correction
        self.correction=correction
        #The hyperparameter optimization method
        if hpfitter is None:
            from ..hpfitter import HyperparameterFitter
            from ..objectivefunctions.gp import LogLikelihood
            hpfitter=HyperparameterFitter(LogLikelihood())
        self.set_hpfitter(hpfitter)
        # Check if the attributes agree
        self.check_attributes()
        #Set hyperparameters
        self.hp={'noise':np.array([-8.0]),'prefactor':np.array([0.0])}
        self.set_hyperparams(hp)
    
    def train(self,features,targets,**kwargs):
        """Train the Gaussian process with training features and targets. 
        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array
                Training targets with N data points 
            or 
            targets: (N,1+D) array
                Training targets in first column and derivatives of each feature in the next columns if use_derivatives is True
        Returns trained Gaussian process:
        """
        #Make kernel matrix with noise
        self.features=features.copy()
        K=self.kernel(features,get_derivatives=self.use_derivatives)
        K=self.add_regularization(K,len(features))
        self.L,self.low=cho_factor(K)
        #Subtracting prior mean from target 
        targets=targets.copy()
        self.prior.update(features,targets,L=self.L,low=self.low)
        targets=targets-self.prior.get(features,targets,get_derivatives=self.use_derivatives)
        #Rearrange targets if derivatives are used
        if self.use_derivatives:
            targets=targets.T.reshape(-1,1)
        else:
            targets=targets[:,0:1]
        #Calculate the coefficients
        self.coef=cho_solve((self.L,self.low),targets,check_finite=False)
        return self

    def predict(self,features,get_variance=False,get_derivatives=False,include_noise=False,**kwargs):
        """Predict the mean and variance for test features by using data and coefficients from training data.
        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            get_variance: bool
                Whether to predict the variance
            get_derivatives: bool
                Whether to predict the derivative mean and uncertainty
            include_noise: bool
                Whether to include the noise of data in the predicted variance
        Returns:
            Y_predict: (M,1) or (M,1+D) array 
                The predicted mean values and or without derivatives
            var: (M,1) or (M,1+D) array
                The predicted variance of values and or without derivatives.
        """
        #Calculate the kernel matrix of test and training data
        KQX=self.kernel(features,self.features,get_derivatives=get_derivatives)
        n_data=len(features)
        #Calculate the predicted values
        Y_predict=np.matmul(KQX,self.coef)
        Y_predict=Y_predict.reshape(n_data,-1,order='F')
        Y_predict=Y_predict+self.prior.get(features,Y_predict,get_derivatives=get_derivatives)
        #Calculate the predicted variance
        if get_variance:
            var=self.calculate_variance(features,KQX,get_derivatives=get_derivatives,include_noise=include_noise)
            return Y_predict,var
        return Y_predict,None

    def calculate_variance(self,features,KQX,get_derivatives=False,include_noise=False,**kwargs):
        """Calculate the predicted variance
        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) array or (M*(1+D),N*(1+D)) array or (M,N*(1+D))
                The kernel matrix of test and training data.
            get_derivatives: bool
                Whether to predict the derivative uncertainty.
            include_noise: bool
                Whether to include the noise of data in the predicted variance
        Returns:
            var: (M,1) or (M,1+D) array
                The predicted variance of values and or without derivatives.
        """
        #Calculate the diagonal elements of the kernel matrix without noise 
        n_data=len(features)
        k=self.kernel.diag(features,get_derivatives=get_derivatives)
        if include_noise:
            if get_derivatives and 'noise_deriv' in self.hp:
                k[range(n_data)]+=(np.nan_to_num(np.exp(2*self.hp['noise'][0]))+self.corr)
                k[range(n_data,len(k))]+=(np.nan_to_num(np.exp(2*self.hp['noise_deriv'][0]))+self.corr)
            else:
                k+=(np.nan_to_num(np.exp(2*self.hp['noise'][0]))+self.corr)
        #Calculate predicted variance
        var=(k-np.einsum('ij,ji->i',KQX,cho_solve((self.L,self.low),KQX.T,check_finite=False))).reshape(-1,1)
        var=var*np.exp(2*self.hp['prefactor'][0])
        if get_derivatives:
            return var.reshape(n_data,-1,order='F')
        return var

    def set_hyperparams(self,new_params,**kwargs):
        """Set or update the hyperparameters for the model.
            Parameters:
                new_params: dictionary
                    A dictionary of hyperparameters that are added or updated.
            Returns:
                hp: dictionary
                    An updated dictionary of hyperparameters with prefactor, noise, kernel hyperparameters (like length) 
                    and noise_deriv for the derivative part of the kernel if specified.
        """
        self.kernel.set_hyperparams(new_params)
        # Prefactor and noise is always in the GP
        if 'prefactor' in new_params:
            self.hp['prefactor']=np.array(new_params['prefactor'],dtype=float).reshape(-1)
        if 'noise' in new_params:
            self.hp['noise']=np.array(new_params['noise'],dtype=float).reshape(-1)
        if 'noise_deriv' in new_params:
            self.hp['noise_deriv']=np.array(new_params['noise_deriv'],dtype=float).reshape(-1)
        return self

    def get_gradients(self,X,hp,KXX,**kwargs):
        """Get the gradients of the Gaussian Process with respect to the hyperparameters.
            Parameters:
                X: (N,D) array
                    Features with N data points and D dimensions.
                hp: list
                    A list with elements of the hyperparameters that are optimized.
                KXX: (N,N) array
                    The kernel matrix of training data.
        """
        hp_deriv={}
        n_data,m_data=len(X),len(KXX)
        if 'prefactor' in hp:
            hp_deriv['prefactor']=np.array([2*np.exp(2*self.hp['prefactor'][0])*self.add_regularization(KXX,n_data,overwrite=False)])
        if 'noise' in hp:
            if 'noise_deriv' in self.hp:
                hp_deriv['noise']=np.array([np.diag(np.array([2.0*np.exp(2.0*self.hp['noise'][0])]*n_data+[0.0]*int(m_data-n_data)).reshape(-1))])
            else:
                hp_deriv['noise']=np.array([np.diag(np.array([2.0*np.exp(2.0*self.hp['noise'][0])]*m_data).reshape(-1))])
        if 'noise_deriv' in hp:
            hp_deriv['noise_deriv']=np.array([np.diag(np.array([0.0]*n_data+[2*np.exp(2*self.hp['noise_deriv'][0])]*int(m_data-n_data)).reshape(-1))])
        hp_deriv.update(self.kernel.get_gradients(X,hp,KXX=KXX))
        return hp_deriv
    
    def copy(self):
        " Copy the Model object. "
        clone=self.__class__(prior=self.prior,
                             kernel=self.kernel,
                             hp=self.hp,
                             use_derivatives=self.use_derivatives,
                             correction=self.correction,
                             hpfitter=self.hpfitter)
        # Inherit floats or bools
        for key in ['corr','low']:
            if key in self.__dict__.keys():
                clone.__dict__[key]=self.__dict__[key]
        # Inherit ndarrays
        for key in ['features','L','coef']:
            if key in self.__dict__.keys():
                clone.__dict__[key]=self.__dict__[key].copy()
        return clone

    def __repr__(self):
        return "GaussianProcess(prior={}, kernel={}, hp={}, use_derivatives={}, correction={}, hpfitter={})".format(self.prior,self.kernel,self.hp,self.use_derivatives,self.correction,self.hpfitter)

