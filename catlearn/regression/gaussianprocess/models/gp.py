import numpy as np
from .model import ModelProcess

class GaussianProcess(ModelProcess):
    def __init__(self,prior=None,kernel=None,hpfitter=None,hp={},use_derivatives=False,use_correction=True,**kwargs):
        """
        The Gaussian Process Regressor.
        The Gaussian process uses Cholesky decomposition for inverting the kernel matrix.
        The hyperparameters can be optimized.
        Parameters:
            prior: Prior class
                The prior given for new data.
            kernel: Kernel class
                The kernel function used for the kernel matrix.
            hpfitter: HyperparameterFitter class
                A class to optimize hyperparameters
            hp: dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives: bool
                Use derivatives/gradients for training and predictions.
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
        # Set default hyperparameters
        self.hp={'noise':np.array([-8.0]),'prefactor':np.array([0.0])}
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

    def set_hyperparams(self,new_params,**kwargs):
        self.kernel.set_hyperparams(new_params)
        # Prefactor and relative-noise hyperparameter is always in the GP
        if 'prefactor' in new_params:
            self.hp['prefactor']=np.array(new_params['prefactor'],dtype=float).reshape(-1)
            self.prefactor=self.calculate_prefactor()
        if 'noise' in new_params:
            self.hp['noise']=np.array(new_params['noise'],dtype=float).reshape(-1)
        if 'noise_deriv' in new_params:
            self.hp['noise_deriv']=np.array(new_params['noise_deriv'],dtype=float).reshape(-1)
        return self

    def get_gradients(self,features,hp,KXX,**kwargs):
        hp_deriv={}
        n_data,m_data=len(features),len(KXX)
        if 'prefactor' in hp:
            hp_deriv['prefactor']=np.array([2.0*np.exp(2.0*self.hp['prefactor'][0])*self.add_regularization(KXX,n_data,overwrite=False)])
        if 'noise' in hp:
            K_deriv=np.full(m_data,2.0*np.exp(2.0*self.hp['noise'][0]))
            if 'noise_deriv' in self.hp:
                K_deriv[n_data:]=0.0
                hp_deriv['noise']=np.array([np.diag(K_deriv)])
            else:
                hp_deriv['noise']=np.array([np.diag(K_deriv)])
        if 'noise_deriv' in hp:
            K_deriv=np.full(m_data,2.0*np.exp(2.0*self.hp['noise_deriv'][0]))
            K_deriv[:n_data]=0.0
            hp_deriv['noise_deriv']=np.array([np.diag(K_deriv)])
        hp_deriv.update(self.kernel.get_gradients(features,hp,KXX=KXX))
        return hp_deriv
    
    def calculate_prefactor(self,features=None,targets=None,**kwargs):
        " Calculate the prefactor that the prediction uncertainty is scaled with. "
        return np.exp(2.0*self.hp['prefactor'][0])
