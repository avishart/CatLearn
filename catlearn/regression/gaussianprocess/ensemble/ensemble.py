import numpy as np

class EnsembleModel:
    def __init__(self,model=None,variance_ensemble=True,same_prior=True,**kwargs):
        " Ensemble model of machine learning models. "
        #if model is None:
        #    model
        self.model=model.copy()
        self.variance_ensemble=variance_ensemble
        self.same_prior=same_prior
        
    def train(self,features,targets,**kwargs):
        """Train the model with training features and targets. 
        Parameters:
            features : (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets : (N,1) array
                Training targets with N data points 
            or 
            targets : (N,1+D) array
                Training targets in first column and derivatives of each feature in the next columns if use_derivatives is True
        Returns trained model:
        """
        raise NotImplementedError()
    
    def predict(self,features,get_variance=False,get_derivatives=False,include_noise=False,**kwargs):
        """Predict the mean and variance for test features by using data and coefficients from training data.
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
        #Calculate the predicted values
        Y_predictions=[]
        var_predictions=[]
        if self.n_models==1:
            return self.model.predict(features,get_variance=get_variance,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)
        for model in self.models:
            if self.variance_ensemble or get_variance:
                Y_predict,var=model.predict(features,get_variance=True,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)
            else:
                Y_predict,var=model.predict(features,get_variance=False,get_derivatives=get_derivatives,include_noise=include_noise,**kwargs)
            Y_predictions.append(Y_predict)
            var_predictions.append(var)
        return self.ensemble(np.array(Y_predictions),np.array(var_predictions),get_variance=get_variance)
    
    def ensemble(self,Y_predictions,var_predictions,get_variance=True,**kwargs):
        " Make an ensemble of the predicted values and variances. The variance weighted ensemble is used if variance_ensemble=True. "
        var_predict=None
        if self.variance_ensemble:
            var_coef=1.0/var_predictions
            var_coef=var_coef/np.sum(var_coef,axis=0)
            Y_predict=np.sum(Y_predictions*var_coef,axis=0)
            if get_variance:
                var_predict=np.sum(var_predictions*var_coef,axis=0)+np.sum(var_coef*(Y_predictions-Y_predict)**2,axis=0)
        else:
            Y_predict=np.mean(Y_predictions,axis=0)
            if get_variance:
                var_predict=np.mean(var_predictions,axis=0)+np.mean((Y_predictions-Y_predict)**2,axis=0)
        return Y_predict,var_predict
            
    def optimize(self,features,targets,retrain=True,hp=None,pdis=None,verbose=False,**kwargs):
        """ Optimize the hyperparameter of the model and its kernel
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
        """
        raise NotImplementedError()
        
    def update_model(self,model):
        " Update the default model used. "
        self.model=model.copy()
        return self        
    
    def copy(self):
        " Copy the Model object. "
        clone=self.__class__(model=self.model,
                             variance_ensemble=self.variance_ensemble,
                             same_prior=self.same_prior)
        if 'n_models' in self.__dict__.keys():
            clone.n_models=self.n_models
        if 'models' in self.__dict__.keys():
            clone.models=[model.copy() for model in self.models]
        return clone

    def __repr__(self):
        return "EnsembleModel(model={},variance_ensemble={},same_prior={})".format(self.model,self.hp,self.variance_ensemble,self.same_prior)
