import numpy as np
from scipy.linalg import cho_factor,cho_solve

class ModelProcess:
    def __init__(self,prior=None,kernel=None,hpfitter=None,hp={},use_derivatives=False,use_correction=True,**kwargs):
        """
        The Model Process Regressor.
        The Model process uses Cholesky decomposition for inverting the kernel matrix.
        The hyperparameters can be optimized.

        Parameters:
            prior : Prior class
                The prior mean given for the data.
            kernel : Kernel class
                The kernel function used for the kernel matrix.
            hpfitter : HyperparameterFitter class
                A class to optimize hyperparameters
            hp : dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives : bool
                Use derivatives/gradients of the targets for training and predictions.
            use_correction : bool
                Use the noise correction on the covariance matrix.
        """
        # Set default descriptors
        self.trained_model=False
        self.corr=0.0
        self.features=[]
        self.L=np.array([])
        self.low=False
        self.coef=np.array([])
        self.prefactor=1.0
        # Set default relative-noise hyperparameter
        self.hp={'noise':np.array([-8.0])}
        # Set the default prior mean class
        if prior is None:
            from ..means.mean import Prior_mean
            prior=Prior_mean()
        # Set the default kernel class
        if kernel is None:
            from ..kernel import SE
            kernel=SE(use_derivatives=use_derivatives,use_fingerprint=False)
        # The default hyperparameter optimization method
        if hpfitter is None:
            from ..hpfitter import HyperparameterFitter
            from ..objectivefunctions.gp.likelihood import LogLikelihood
            hpfitter=HyperparameterFitter(func=LogLikelihood())
        # Set all the arguments
        self.update_arguments(prior=prior,
                              kernel=kernel,
                              hpfitter=hpfitter,
                              hp=hp,
                              use_derivatives=use_derivatives,
                              use_correction=use_correction,
                              **kwargs)
    
    def train(self,features,targets,**kwargs):
        """
        Train the model with training features and targets. 

        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array or (N,1+D) array
                Training targets with N data points.
                If use_derivatives=True, the training targets is in first column and derivatives is in the next columns.
        
        Returns:
            self: The trained object itself.
        """
        # Note that the model is trained
        self.trained_model=True
        # Store features
        self.features=features.copy()
        # Make the kernel matrix decomposition
        self.L,self.low=self.calculate_kernel_decomposition(features)
        # Modify the targets with the prior mean and rearrangement
        targets=self.modify_targets(features,targets)
        # Calculate the coefficients
        self.coef=self.calculate_coefficients(features,targets)
        # Calculate the prefactor for variance predictions
        self.prefactor=self.calculate_prefactor(features,targets)
        return self
    
    def optimize(self,features,targets,retrain=True,hp=None,pdis=None,verbose=False,**kwargs):
        """ 
        Optimize the hyperparameter of the model and its kernel.

        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            retrain : bool
                Whether to retrain the model after the optimization.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
                The hyperparameters are used in the log-space.
            maxiter : int
                Maximum number of iterations used by local or global optimization method.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
            verbose : bool
                Print the optimized hyperparameters and the object function value.
                
        Returns:
            dict: A solution dictionary with objective function value and hyperparameters.
        """
        # Ensure the targets are in the right format
        if not self.use_derivatives:
            targets=targets[:,0:1].copy()
        # Optimize the hyperparameters
        sol=self.hpfitter.fit(features,targets,model=self,hp=hp,pdis=pdis,**kwargs)
        # Print the solution
        if verbose:
            print(sol)
        # Retrain the model with the new hyperparameters
        if retrain:
            if 'prior' in sol.keys():
                self.prior.update_arguments(sol['prior'])
            self.set_hyperparams(sol['hp'])
            self.train(features,targets)
        return sol

    def predict(self,features,get_derivatives=False,get_variance=False,include_noise=False,get_derivtives_var=False,get_var_derivatives=False,**kwargs):
        """
        Predict the mean and variance for test features by using data and coefficients from training data.

        Parameters:
            features : (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            get_derivatives : bool
                Whether to predict the derivatives of the prediction mean.
            get_variance : bool
                Whether to predict the variance of the targets.
            include_noise : bool
                Whether to include the noise of data in the predicted variance.
            get_derivtives_var : bool
                Whether to predict the variance of the derivatives of the targets.
            get_var_derivatives : bool
                Whether to calculate the derivatives of the predicted variance of the targets.

        Returns:
            Y_predict : (M,1) or (M,1+D) array 
                The predicted mean values with or without derivatives.
            var : (M,1) or (M,1+D) array
                The predicted variance of the targets with or without derivatives.
            var_deriv : (M,D) array
                The derivatives of the predicted variance of the targets.
        """
        # Check if the model is trained
        if not self.trained_model:
            raise Exception('The model is not trained!')
        # Calculate the kernel matrix of test and training data
        if get_derivatives or (get_derivtives_var and get_variance) or get_var_derivatives:
            KQX=self.kernel(features,self.features,get_derivatives=True)
        else:
            KQX=self.kernel(features,self.features,get_derivatives=False) 
        # Calculate the prediction mean
        Y_predict=self.predict_mean(features,KQX=KQX,get_derivatives=get_derivatives)
        # Calculate the predicted variance
        if get_variance:
            var=self.predict_variance(features,KQX=KQX,get_derivatives=get_derivtives_var,include_noise=include_noise)
        else:
            var=None
        # Calculate the derivatives of the predicted variance
        if get_var_derivatives:
            var_deriv=self.calculate_variance_derivatives(features,KQX=KQX)
        else:
            var_deriv=None
        return Y_predict,var,var_deriv

    def predict_mean(self,features,KQX=None,get_derivatives=False,**kwargs):
        """
        Predict the mean for test features by using data and coefficients from training data.

        Parameters:
            features : (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX : (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features. 
                If KQX=None, it is calculated.
            get_derivatives : bool
                Whether to predict the derivatives of the prediction mean.

        Returns:
            Y_predict : (M,1) array
                The predicted mean values if get_derivatives=False
            or 
            Y_predict : (M,1+D) array 
                The predicted mean values and its derivatives if get_derivatives=True
        """
        # Check if the model is trained
        if not self.trained_model:
            raise Exception('The model is not trained!')
        # Get the number of test points
        m_data=len(features)
        # Calculate the kernel matrix of test and training data if it is not given
        if KQX is None:
            KQX=self.kernel(features,self.features,get_derivatives=get_derivatives)
        else:
            if not get_derivatives:
                KQX=KQX[:m_data]
        # Calculate the prediction mean
        Y_predict=np.matmul(KQX,self.coef)
        # Rearrange prediction
        Y_predict=Y_predict.reshape(m_data,-1,order='F')
        # Add the prior mean 
        Y_predict=Y_predict+self.prior.get(features,Y_predict,get_derivatives=get_derivatives)
        return Y_predict
    
    def predict_variance(self,features,KQX=None,get_derivatives=False,include_noise=False,**kwargs):
        """
        Calculate the predicted variance of the test targets.

        Parameters:
            features : (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX : (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features. 
                If KQX=None, it is calculated.
            get_derivatives : bool
                Whether to predict the uncertainty of the derivatives of the targets.
            include_noise : bool
                Whether to include the noise of data in the predicted variance

        Returns:
            var : (M,1) array
                The predicted variance of the targets if get_derivatives=False.
            or 
            var : (M,1+D) array
                The predicted variance of the targets and its derivatives if get_derivatives=True.

        """
        # Check if the model is trained
        if not self.trained_model:
            raise Exception('The model is not trained!')
        # Get the number of test points
        m_data=len(features)
        # Calculate the kernel matrix of test and training data if it is not given
        if KQX is None:
            KQX=self.kernel(features,self.features,get_derivatives=get_derivatives)
        else:
            if not get_derivatives:
                KQX=KQX[:m_data]
        # Calculate the diagonal elements of the kernel matrix of the test data 
        k=self.kernel_diag(features,m_data=m_data,get_derivatives=get_derivatives,include_noise=include_noise)
        # Calculate predicted variance
        var=(k-np.einsum('ij,ji->i',KQX,self.calculate_CinvKQX(KQX))).reshape(-1,1)
        # Scale prediction variance with the prefactor
        var=var*self.prefactor
        # Rearrange the predicted variance
        return var.reshape(m_data,-1,order='F')
    
    def calculate_variance_derivatives(self,features,KQX=None,**kwargs):
        """
        Calculate the derivatives of the predicted variance of the test targets.

        Parameters:
            features : (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX : (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features. 
                If KQX=None, it is calculated.

        Returns:
            var_deriv : (M,D) array
                The derivatives of the predicted variance of the targets.
        """
        # Check if the model is trained
        if not self.trained_model:
            raise Exception('The model is not trained!')
        # Get the number of test points
        m_data=len(features)
        # Calculate the kernel matrix of test and training data
        if KQX is None:
            KQX=self.kernel(features,self.features,get_derivatives=True)
        # Calculate the derivative of the diagonal elements wrt. the test features
        k_deriv=self.kernel_deriv_diag(features)
        # Calculate derivative of the predicted variance
        var_deriv=k_deriv-2.0*np.einsum('ij,ji->i',KQX[m_data:],self.calculate_CinvKQX(KQX[:m_data])).reshape(-1,1)
        # Scale prediction variance with the prefactor
        var_deriv=var_deriv*self.prefactor
        # Rearrange derivative of variance
        return var_deriv.reshape(m_data,-1,order='F')
    
    def set_hyperparams(self,new_params,**kwargs):
        """
        Set or update the hyperparameters for the model.
        Parameters:
            new_params : dictionary
                A dictionary of hyperparameters that are added or updated.
                The hyperparameters are used in the log-space.
        Returns:
            self: The object itself with the new hyperparameters.
        """
        self.kernel.set_hyperparams(new_params)
        if 'noise' in new_params:
            self.hp['noise']=np.array(new_params['noise'],dtype=float).reshape(-1)
        if 'noise_deriv' in new_params:
            self.hp['noise_deriv']=np.array(new_params['noise_deriv'],dtype=float).reshape(-1)
        return self
    
    def get_hyperparams(self,**kwargs):
        """
        Get the hyperparameters for the model and the kernel.
        Returns:
            dict: The hyperparameters in the log-space from the model and kernel class.  
        """
        hp={para:value.copy() for para,value in self.hp.items()}
        hp.update(self.kernel.get_hyperparams())
        return hp
    
    def get_gradients(self,X,hp,KXX,**kwargs):
        """
        Get the gradients of the covariance matrix with noise wrt. the hyperparameters.
        Parameters:
            X : (N,D) array
                Features with N data points and D dimensions.
            hp : list
                A list with elements of the hyperparameters that are optimized.
            KXX : (N,N) array
                The kernel matrix of training data.
        Returns:
            dict: A dictionary with gradients of the covariance matrix with noise wrt. the hyperparameters. 
        """
        raise NotImplementedError()
    
    def update_arguments(self,prior=None,kernel=None,hpfitter=None,hp={},use_derivatives=None,use_correction=None,**kwargs):
        """
        Update the Model Process Regressor with its arguments. The existing arguments are used if they are not given.
        Parameters:
            prior : Prior class
                The prior given for new data.
            kernel : Kernel class
                The kernel function used for the kernel matrix.
            hpfitter : HyperparameterFitter class
                A class to optimize hyperparameters
            hp : dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives : bool
                Use derivatives/gradients for training and predictions.
            use_correction : bool
                Use the noise correction on the covariance matrix.
        Returns:
            self: The updated object itself.
        """
        # Set the prior mean class
        if prior is not None:
            self.prior=prior.copy()
        # Set the kernel class
        if kernel is not None:
            self.kernel=kernel.copy()
        # Set whether to use derivatives for the target
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        # Set noise correction
        if use_correction is not None:
            self.use_correction=use_correction
        # The hyperparameter optimization method
        if hpfitter is not None:
            self.hpfitter=hpfitter.copy()
        # Set hyperparameters
        self.set_hyperparams(hp)
        # Check if the attributes agree
        self.check_attributes()
        return self

    def add_regularization(self,K,n_data,overwrite=True,**kwargs):
        """
        Add the regularization to the diagonal elements of the squared kernel matrix. 
        (K will be overwritten if overwrite=True)
        """
        # Whether to make a copy of the kernel matrix
        if not overwrite:
            K=K.copy()
        m_data=len(K)
        #Calculate the correction, so the kernel matrix is invertible
        self.corr=self.get_correction(np.diag(K))
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
        
    def get_correction(self,K_diag,**kwargs):
        " Get the noise correction, so that the training covariance matrix is always invertible. "
        if self.use_correction:
            return (np.sum(K_diag)**2)*(1.0/(1.0/(2.3e-16)-(len(K_diag)**2)))
        return 0.0
    
    def calculate_kernel_decomposition(self,features,**kwargs):
        " Do the Cholesky decomposition of the kernel matrix. "
        # Make kernel matrix with noise
        K=self.kernel(features,get_derivatives=self.use_derivatives)
        K=self.add_regularization(K,len(features))
        # Do Cholesky decomposition
        return cho_factor(K)
    
    def modify_targets(self,features,targets,**kwargs):
        " Modify the targets with the prior mean and rearrangement. "
        # Subtracting prior mean from target 
        targets=targets.copy()
        self.prior.update(features,targets,L=self.L,low=self.low)
        targets=targets-self.prior.get(features,targets,get_derivatives=self.use_derivatives)
        # Rearrange targets if derivatives are used
        if self.use_derivatives:
            targets=targets.T.reshape(-1,1)
        else:
            targets=targets[:,0:1].copy()
        return targets
    
    def calculate_coefficients(self,features,targets,**kwargs):
        " Calculate the coefficients for the prediction mean. "
        return cho_solve((self.L,self.low),targets,check_finite=False)
    
    def calculate_prefactor(self,features,targets,**kwargs):
        " Calculate the prefactor that the prediction uncertainty is scaled with. "
        raise NotImplementedError()

    def kernel_diag(self,features,m_data,get_derivatives=False,include_noise=False,**kwargs):
        " Calculate the diagonal elements of the kernel matrix of the test data. "
        # Calculate the diagonal elements of the kernel matrix of the test data without noise 
        k=self.kernel.diag(features,get_derivatives=get_derivatives)
        # Add noise to the kernel elements
        if include_noise:
            if get_derivatives and 'noise_deriv' in self.hp:
                k[:m_data]+=(np.nan_to_num(np.exp(2.0*self.hp['noise'][0]))+self.corr)
                k[m_data:]+=(np.nan_to_num(np.exp(2.0*self.hp['noise_deriv'][0]))+self.corr)
            else:
                k+=(np.nan_to_num(np.exp(2.0*self.hp['noise'][0]))+self.corr)
        return k
    
    def kernel_deriv_diag(self,features,**kwargs):
        " Calculate the derivative of the diagonal elements in the kernel wrt. the features. "
        return self.kernel.diag_deriv(features,**kwargs)
    
    def calculate_CinvKQX(self,KQX,**kwargs):
        " Calculate the CinvKQX matrix. "
        return cho_solve((self.L,self.low),KQX.T,check_finite=False)

    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.use_derivatives!=self.kernel.use_derivatives:
            raise Exception('The Model and the Kernel do not agree whether to use derivatives!')
        return True

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(prior=self.prior,
                        kernel=self.kernel,
                        hpfitter=self.hpfitter,
                        hp=self.get_hyperparams(),
                        use_derivatives=self.use_derivatives,
                        use_correction=self.use_correction)
        # Get the constants made within the class
        constant_kwargs=dict(trained_model=self.trained_model,
                             corr=self.corr,
                             low=self.low,
                             prefactor=self.prefactor)
        # Get the objects made within the class
        object_kwargs=dict(features=self.features,
                           L=self.L,
                           coef=self.coef)
        return arg_kwargs,constant_kwargs,object_kwargs

    def copy(self):
        " Copy the object. "
        # Get all arguments
        arg_kwargs,constant_kwargs,object_kwargs=self.get_arguments()
        # Make a clone
        clone=self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key,value in constant_kwargs.items():
                clone.__dict__[key]=value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key,value in object_kwargs.items():
                clone.__dict__[key]=value.copy()
        return clone
    
    def __repr__(self):
        arg_kwargs=self.get_arguments()[0]
        str_kwargs=",".join([f"{key}={value}" for key,value in arg_kwargs.items()])
        return "{}({})".format(self.__class__.__name__,str_kwargs)