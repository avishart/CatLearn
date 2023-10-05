import numpy as np
from scipy.spatial.distance import pdist
from .restricted import RestrictedBoundaries

class EducatedBoundaries(RestrictedBoundaries):
    def __init__(self,bounds_dict={},scale=1.0,log=True,max_length=True,use_derivatives=False,use_prior_mean=True,**kwargs):
        """ 
        Boundary conditions for the hyperparameters with educated guess for 
        the length-scale, relative-noise, and prefactor hyperparameters.
        Machine precisions are used as boundary conditions for other hyperparameters not given in the dictionary.
        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
            max_length : bool
                Whether to use the maximum scaling for the length-scale or use a more reasonable scaling.
            use_derivatives : bool
                Whether the derivatives of the target are used in the model. 
                The boundary conditions of the length-scale hyperparameter(s) will change with the use_derivatives. 
                The use_derivatives will be updated when update_bounds is called.
            use_prior_mean : bool
                Whether to use the prior mean to calculate the boundary of the prefactor hyperparameter.
                If use_prior_mean=False the minimum and maximum target differences are used as the boundary conditions. 
        """
        self.update_arguments(bounds_dict=bounds_dict,
                              scale=scale,
                              log=log,
                              max_length=max_length,
                              use_derivatives=use_derivatives,
                              use_prior_mean=use_prior_mean,
                              **kwargs)

    def update_arguments(self,bounds_dict=None,scale=None,log=None,max_length=None,use_derivatives=None,use_prior_mean=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.
        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
            max_length : bool
                Whether to use the maximum scaling for the length-scale or use a more reasonable scaling.
            use_derivatives : bool
                Whether the derivatives of the target are used in the model. 
                The boundary conditions of the length-scale hyperparameter(s) will change with the use_derivatives. 
                The use_derivatives will be updated when update_bounds is called.
            use_prior_mean : bool
                Whether to use the prior mean to calculate the boundary of the prefactor hyperparameter.
                If use_prior_mean=False the minimum and maximum target differences are used as the boundary conditions. 
        Returns:
            self: The updated object itself.
        """
        if bounds_dict is not None:
            self.initiate_bounds_dict(bounds_dict)
        if scale is not None:
            self.scale=scale
        if log is not None:
            self.log=log
        if max_length is not None:
            self.max_length=max_length
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        if use_prior_mean is not None:
            self.use_prior_mean=use_prior_mean
        return self
    
    def make_bounds(self,model,X,Y,parameters,parameters_set,**kwargs):
        " Make the boundary conditions with educated guesses of the length-scale hyperparamer(s). "
        eps_lower,eps_upper=self.get_boundary_limits()
        self.get_use_derivatives(model)
        bounds={}
        for para in parameters_set:
            if para=='length':
                bounds[para]=self.length_bound(X,parameters.count(para))
            elif para=='noise':
                if 'noise_deriv' in parameters_set:
                    bounds[para]=self.noise_bound(Y[:,0:1],eps_lower=eps_lower)
                else:
                    bounds[para]=self.noise_bound(Y,eps_lower=eps_lower)
            elif para=='noise_deriv':
                bounds[para]=self.noise_bound(Y[:,1:],eps_lower=eps_lower)
            elif para=='prefactor':
                bounds[para]=self.prefactor_bound(X,Y,model)
            elif para in self.bounds_dict:
                bounds[para]=self.bounds_dict[para].copy()
            else:
                bounds[para]=np.full((parameters.count(para),2),[eps_lower,eps_upper])
        return bounds
    
    def prefactor_bound(self,X,Y,model,**kwargs):
        " Get the minimum and maximum ranges of the prefactor in the educated guess regime within a scale. "
        if self.use_prior_mean:
            # Get the prior mean value for the target only (without derivatives)
            Y_mean=self.get_prior_mean(X,Y,model)
            Y_std=Y[:,0:1]-Y_mean
            # Calculate the variance relative to the prior mean of the targets
            a_mean=np.sqrt(np.mean(Y_std**2))
            # Check that all the targets are not the same
            if a_mean==0.0:
                a_mean=1.00
            scaling=10.0*self.scale
            a_max=a_mean*scaling
            a_min=a_mean/scaling
        else:
            # Calculate the differences in the target values
            dif=pdist(Y[:,0:1])
            # Remove zero differences
            dif=dif[dif!=0.0]
            # Check that all the targets are not the same
            if len(dif)==0:
                dif=[1.0]
            a_max=np.max(dif)*self.scale
            a_min=np.min(dif)/self.scale
        if self.log:
            return np.array([[np.log(a_min),np.log(a_max)]])
        return np.array([[a_min,a_max]])
    
    def get_prior_mean(self,X,Y,model,**kwargs):
        " Get the prior mean value for the target only (without derivatives). "
        # Update the prior mean used in ML model
        model.prior.update(X,Y)
        return model.prior.get(X,Y[:,0:1],get_derivatives=False)
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(bounds_dict=self.bounds_dict,
                        scale=self.scale,
                        log=self.log,
                        max_length=self.max_length,
                        use_derivatives=self.use_derivatives,
                        use_prior_mean=self.use_prior_mean)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs
    