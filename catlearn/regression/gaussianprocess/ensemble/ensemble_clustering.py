import numpy as np
from .ensemble import EnsembleModel
from ..means.constant import Prior_constant

class EnsembleClustering(EnsembleModel):
    def __init__(self,model=None,clustering=None,variance_ensemble=True,same_prior=True,**kwargs):
        " Ensemble model of machine learning models with ensembles from a clustering algorithm. "
        super().__init__(model=model,variance_ensemble=variance_ensemble,same_prior=same_prior)
        #if clustering is None:
        #    clustering
        self.clustering=clustering.copy()
        
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
        cdata=self.cluster(features,targets)
        self.n_models=len(cdata)
        self.models=[]
        if self.n_models==1:
            self.model.train(features,targets)
            self.models.append(self.model)
            return self
        if self.same_prior:
            ymean=np.mean(targets[:,0])
        for ki in range(self.n_models):
            model=self.model.copy()
            if self.same_prior:
                model.prior=Prior_constant(yp=ymean)
            model.train(cdata[ki][0],cdata[ki][1],**kwargs)
            self.models.append(model)
        return self
        
    def cluster(self,features,targets,**kwargs):
        " Cluster the data. "
        if isinstance(features[0],(np.ndarray,list)):
            X=features.copy()
        else:
            X=np.array([feature.get_vector() for feature in features])
        cluster_indicies=self.clustering.cluster_fit_data(X)
        return [(features[indicies_ki],targets[indicies_ki]) for indicies_ki in cluster_indicies]
            
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
        cdata=self.cluster(features,targets)
        self.n_models=len(cdata)
        self.models=[]
        sols=[]
        if self.n_models==1:
            sol=self.model.optimize(features,targets,retrain=retrain,hp=hp,pdis=pdis,verbose=verbose,**kwargs)
            sols.append(sol)
            self.models.append(self.model)
            return sols
        if self.same_prior:
            ymean=np.mean(targets[:,0])
        for ki in range(self.n_models):
            model=self.model.copy()
            if self.same_prior:
                model.prior=Prior_constant(yp=ymean)
            sol=model.optimize(cdata[ki][0],cdata[ki][1],retrain=retrain,hp=hp,pdis=pdis,verbose=verbose,**kwargs)
            sols.append(sol)
            self.models.append(model)         
        return sols
    
    def copy(self):
        " Copy the Model object. "
        clone=self.__class__(model=self.model,
                             clustering=self.clustering,
                             variance_ensemble=self.variance_ensemble,
                             same_prior=self.same_prior)
        if 'n_models' in self.__dict__.keys():
            clone.n_models=self.n_models
        if 'models' in self.__dict__.keys():
            clone.models=[model.copy() for model in self.models]
        return clone

    def __repr__(self):
        return "EnsembleClustering(model={},clustering={},variance_ensemble={},same_prior={})".format(self.model,self.clustering,self.variance_ensemble,self.same_prior)
