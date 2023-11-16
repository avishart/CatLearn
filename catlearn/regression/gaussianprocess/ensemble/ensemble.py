import numpy as np

class EnsembleModel:
    def __init__(self,model=None,use_variance_ensemble=True,use_same_prior_mean=True,**kwargs):
        """
        Ensemble model of machine learning models.
        Parameters:
            model : Model
                The Machine Learning Model with kernel and prior that are optimized.
            use_variance_ensemble : bool
                Whether to use the predicted variances to weight the predictions.
                Else an average of the predictions is used.
            use_same_prior_mean : bool
                Whether to use the same prior mean for all models.
        """
        # Make default model if it is not given
        if model is None:
            from ..calculator.mlmodel import get_default_model
            model=get_default_model()
        # Set the arguments
        self.update_arguments(model=model,
                              use_variance_ensemble=use_variance_ensemble,
                              use_same_prior_mean=use_same_prior_mean,
                              **kwargs)
        
    def train(self,features,targets,**kwargs):
        """
        Train the model with training features and targets. 
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array
                Training targets with N data points 
            or 
            targets : (N,1+D) array
                Training targets in first column and derivatives of each feature in the next columns if use_derivatives is True
        Returns:
            self: The trained model object itself.
        """
        raise NotImplementedError()
    
    def predict(self,features,get_variance=False,get_derivatives=False,include_noise=False,**kwargs):
        """
        Predict the mean and variance for test features by using data and coefficients from training data.
        Parameters:
            features : (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            get_variance : bool
                Whether to predict the variance
            get_derivatives : bool
                Whether to predict the derivative mean and uncertainty
            include_noise : bool
                Whether to include the noise of data in the predicted variance
        Returns:
            Y_predict : (M,1) or (M,1+D) array 
                The predicted mean values and or without derivatives
            var : (M,1) or (M,1+D) array
                The predicted variance of values and or without derivatives.
        """
        # Calculate the predicted values for one model
        if self.n_models==1:
            return self.model_prediction(self.model,features,get_variance=get_variance,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)
        #Calculate the predicted values for multiple model
        Y_predictions=[]
        var_predictions=[]
        for model in self.models:
            if self.use_variance_ensemble or get_variance:
                Y_predict,var=self.model_prediction(model,features,get_variance=True,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)
            else:
                Y_predict,var=self.model_prediction(model,features,get_variance=False,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)
            Y_predictions.append(Y_predict)
            var_predictions.append(var)
        return self.ensemble(np.array(Y_predictions),np.array(var_predictions),get_variance=get_variance)
    
    def optimize(self,features,targets,retrain=True,hp=None,pdis=None,verbose=False,**kwargs):
        """ 
        Optimize the hyperparameter of the model and its kernel
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            retrain : bool
                Whether to retrain the model after the optimization.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            maxiter : int
                Maximum number of iterations used by local or global optimization method.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
            verbose : bool
                Print the optimized hyperparameters and the object function value.
        Returns: 
            list : List of solution dictionaries with objective function value, optimized hyperparameters,
                success statement, and number of used evaluations.
        """
        raise NotImplementedError()
    
    def update_arguments(self,model=None,use_variance_ensemble=None,use_same_prior_mean=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            model : Model
                The Machine Learning Model with kernel and prior that are optimized.
            use_variance_ensemble : bool
                Whether to use the predicted variances to weight the predictions.
                Else an average of the predictions is used.
            use_same_prior_mean : bool
                Whether to use the same prior mean for all models.
        Returns:
            self: The updated object itself.
        """
        if model is not None:
            self.model=model.copy()
            # Set descriptor of the ensemble model
            self.n_models=1
            self.models=[]
        if use_variance_ensemble is not None:
            self.use_variance_ensemble=use_variance_ensemble
        if use_same_prior_mean is not None:
            self.use_same_prior_mean=use_same_prior_mean
        return self
    
    def model_training(self,model,features,targets,**kwargs):
        " Train the model. "
        return model.train(features,targets,**kwargs)
    
    def model_optimization(self,model,features,targets,retrain=True,hp=None,pdis=None,verbose=False,**kwargs):
        " Optimize the hyperparameters of the model. "
        return model.optimize(features,targets,retrain=retrain,hp=hp,pdis=pdis,verbose=verbose,**kwargs)
    
    def model_prediction(self,model,features,get_variance=False,get_derivatives=False,include_noise=False,**kwargs):
        " Predict mean and variance with the model. "
        return model.predict(features,get_variance=get_variance,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)

    def ensemble(self,Y_predictions,var_predictions,get_variance=True,**kwargs):
        " Make an ensemble of the predicted values and variances. The variance weighted ensemble is used if variance_ensemble=True. "
        var_predict=None
        if self.use_variance_ensemble:
            # Use the predicted variance to weight predictions
            var_coef=1.0/var_predictions
            var_coef=var_coef/np.sum(var_coef,axis=0)
            Y_predict=np.sum(Y_predictions*var_coef,axis=0)
            if get_variance:
                var_predict=np.sum(var_predictions*var_coef,axis=0)+np.sum(var_coef*(Y_predictions-Y_predict)**2,axis=0)
        else:
            # Use the average to weight predictions
            Y_predict=np.mean(Y_predictions,axis=0)
            if get_variance:
                var_predict=np.mean(var_predictions,axis=0)+np.mean((Y_predictions-Y_predict)**2,axis=0)
        return Y_predict,var_predict
    
    def get_models(self,**kwargs):
        " Get the models. "
        return [model.copy() for model in self.models]
    
    def set_same_prior_mean(self,model,features,targets,**kwargs):
        " Set the same prior mean constant for the models. "
        if self.use_same_prior_mean:
            from ..means.constant import Prior_constant
            ymean=np.mean(targets[:,0])
            model.update_arguments(prior=Prior_constant(yp=ymean))
        return model
        
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(model=self.model,
                        use_variance_ensemble=self.use_variance_ensemble,
                        use_same_prior_mean=self.use_same_prior_mean)
        # Get the constants made within the class
        constant_kwargs=dict(n_models=self.n_models)
        # Get the objects made within the class
        object_kwargs=dict(models=self.get_models())
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
