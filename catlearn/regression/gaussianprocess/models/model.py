import numpy as np

class ModelProcess:
    def __init__(self,prior=None,kernel=None,hp={},use_derivatives=False,correction=True,hpfitter=None,**kwargs):
        """The Model Process Regressor.
            Parameters:
                prior: Prior class
                    The prior given for new data.
                kernel: Kernel class
                    The kernel function used for the kernel matrix.
                hp: dictionary
                    A dictionary of hyperparameters like noise and length scale.
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
            hpfitter=HyperparameterFitter(None)
        self.set_hpfitter(hpfitter)
        # Check if the attributes agree
        self.check_attributes()
        #Set hyperparameters
        self.hp={'noise':np.array([-8.0])}
        self.set_hyperparams(hp)
    
    def train(self,features,targets,**kwargs):
        """Train the model with training features and targets. 
        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array
                Training targets with N data points 
            or 
            targets: (N,1+D) array
                Training targets in first column and derivatives of each feature in the next columns if use_derivatives is True
        Returns trained model:
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()
    
    def optimize(self,features,targets,retrain=True,hp=None,pdis=None,verbose=False,**kwargs):
        """ Optimize the hyperparameter of the model and its kernel
        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            retrain: bool
                Whether to retrain the model after the optimization.
            hp: dict
                Use a set of hyperparameters to optimize from else the current set is used.
            maxiter: int
                Maximum number of iterations used by local or global optimization method.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            verbose: bool
                Print the optimized hyperparameters and the object function value.
        """
        if not self.use_derivatives:
            targets=targets[:,0:1].copy()
        sol=self.hpfitter.fit(features,targets,model=self,hp=hp,pdis=pdis,**kwargs)
        if verbose:
            print(sol)
        if retrain:
            if 'prior' in sol.keys():
                self.prior.set_parameters(sol['prior'])
            self.set_hyperparams(sol['hp'])
            self.train(features,targets)
        return sol

    def add_regularization(self,K,n_data,overwrite=True,**kwargs):
        "Add the regularization to the diagonal elements of the squared kernel matrix. (K will be overwritten if overwrite=True)"
        #Calculate the correction, so the kernel matrix is invertible
        if not overwrite:
            K=K.copy()
        m_data=len(K)
        self.corr=self.get_correction(np.diag(K)) if self.correction else 0.0
        if 'noise_deriv' in self.hp:
            add_v=self.inf_to_num(np.exp(2*self.hp['noise'][0]))+self.corr
            K[range(n_data),range(n_data)]+=add_v
            add_v=self.inf_to_num(np.exp(2*self.hp['noise_deriv'][0]))+self.corr
            K[range(n_data,m_data),range(n_data,m_data)]+=add_v
        else:
            add_v=self.inf_to_num(np.exp(2*self.hp['noise'][0]))+self.corr
            K[range(m_data),range(m_data)]+=add_v
        return K
    
    def inf_to_num(self,value,replacing=1e300):
        " Check if a value is infinite and then replace it with a large number if it is. "
        if value==np.inf:
            return replacing
        return value

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
        if 'noise' in new_params:
            self.hp['noise']=np.array(new_params['noise'],dtype=float).reshape(-1)
        if 'noise_deriv' in new_params:
            self.hp['noise_deriv']=np.array(new_params['noise_deriv'],dtype=float).reshape(-1)
        return self
    
    def get_hyperparams(self):
        " Get the hyperparameters for the model and the kernel. "
        hp={para:value.copy() for para,value in self.hp.items()}
        hp.update(self.kernel.get_hyperparams())
        return hp

    def set_hpfitter(self,hpfitter):
        " Set the hpfitter "
        self.hpfitter=hpfitter.copy()
        return self
        
    def get_correction(self,K_diag):
        "Get the correction, so that the training covariance matrix is always invertible"
        return (np.sum(K_diag)**2)*(1.0/(1.0/(2.3e-16)-(len(K_diag)**2)))

    def get_gradients(self,X,hp,KXX,**kwargs):
        """Get the gradients of the model with respect to the hyperparameters.
            Parameters:
                X: (N,D) array
                    Features with N data points and D dimensions.
                hp: list
                    A list with elements of the hyperparameters that are optimized.
                KXX: (N,N) array
                    The kernel matrix of training data.
        """
        raise NotImplementedError()

    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.use_derivatives!=self.kernel.use_derivatives:
            raise Exception('The Model and the Kernel do not agree whether to use derivatives!')
        return
    
    def copy(self):
        " Copy the Model object. "
        clone=self.__class__(prior=self.prior,
                             kernel=self.kernel,
                             hp=self.hp,
                             use_derivatives=self.use_derivatives,
                             correction=self.correction,
                             hpfitter=self.hpfitter)
        if 'corr' in self.__dict__.keys():
            clone.corr=self.corr
        return clone

    def __repr__(self):
        return "ModelProcess(prior={}, kernel={}, hp={}, use_derivatives={}, correction={}, hpfitter={})".format(self.prior,self.kernel,self.hp,self.use_derivatives,self.correction,self.hpfitter)

