import numpy as np
from .model import ModelProcess

class TProcess(ModelProcess):
    def __init__(self,prior=None,kernel=None,hpfitter=None,hp={},use_derivatives=False,use_correction=True,a=1e-20,b=1e-20,**kwargs):
        """
        The Student's T Process Regressor.
        The Student's T process uses Cholesky decomposition for inverting the kernel matrix.
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
            a: float
                Hyperprior shape parameter for the inverse-gamma distribution of the prefactor.
            b: float
                Hyperprior scale parameter for the inverse-gamma distribution of the prefactor.
        """
        # Set default descriptors
        self.trained_model=False
        self.corr=0.0
        self.features=[]
        self.L=np.array([])
        self.low=False
        self.coef=np.array([])
        self.prefactor=1.0
        # Set default relative-noise hyperparameters
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
            from ..objectivefunctions.tp.likelihood import LogLikelihood
            hpfitter=HyperparameterFitter(func=LogLikelihood())
        # Set noise hyperparameters
        self.hp={'noise':np.array([-8.0])}
        # Set all the arguments
        self.update_arguments(prior=prior,
                              kernel=kernel,
                              hpfitter=hpfitter,
                              hp=hp,
                              use_derivatives=use_derivatives,
                              use_correction=use_correction,
                              a=a,
                              b=b,
                              **kwargs)

    def set_hyperparams(self,new_params={},**kwargs):
        self.kernel.set_hyperparams(new_params)
        # Noise is always in the TP
        if 'noise' in new_params:
            self.hp['noise']=np.array(new_params['noise'],dtype=float).reshape(-1)
        if 'noise_deriv' in new_params:
            self.hp['noise_deriv']=np.array(new_params['noise_deriv'],dtype=float).reshape(-1)
        return self
    
    def update_arguments(self,prior=None,kernel=None,hpfitter=None,hp={},use_derivatives=None,use_correction=None,a=None,b=None,**kwargs):
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
            a: float
                Hyperprior shape parameter for the inverse-gamma distribution of the prefactor.
            b: float
                Hyperprior scale parameter for the inverse-gamma distribution of the prefactor.
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
        # The hyperprior shape parameter
        if a is not None:
            self.a=float(a)
        # The hyperprior scale parameter
        if b is not None:
            self.b=float(b)
        #Set hyperparameters
        self.set_hyperparams(hp)
        # Check if the attributes agree
        self.check_attributes()
        return self

    def get_gradients(self,features,hp,KXX,**kwargs):
        hp_deriv={}
        n_data,m_data=len(features),len(KXX)
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
    
    def get_hyperprior_parameters(self,**kwargs):
        " Get the hyperprior parameters from the Student's T Process. "
        return self.a,self.b
    
    def calculate_prefactor(self,features,targets,**kwargs):
        " Calculate the prefactor that the prediction uncertainty is scaled with. "
        n2=float(len(targets)-2) if len(targets)>1 else 0.0
        return (2.0*self.b+np.matmul(targets.T,self.coef).item(0))/(2.0*self.a+n2)

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(prior=self.prior,
                        kernel=self.kernel,
                        hpfitter=self.hpfitter,
                        hp=self.get_hyperparams(),
                        use_derivatives=self.use_derivatives,
                        use_correction=self.use_correction,
                        a=self.a,
                        b=self.b)
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
