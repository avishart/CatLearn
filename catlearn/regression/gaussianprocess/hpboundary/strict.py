import numpy as np
from scipy.spatial.distance import pdist
from .educated import EducatedBoundaries

class StrictBoundaries(EducatedBoundaries):
    def __init__(self,bounds_dict={},scale=1.0,log=True,use_derivatives=False,use_prior_mean=True,**kwargs):
        """ 
        Boundary conditions for the hyperparameters with educated guess for 
        the length-scale, relative-noise, and prefactor hyperparameters.
        Stricter boundary conditions are used for the length-scale hyperparameter.
        Machine precisions are used as boundary conditions for other hyperparameters not given in the dictionary.        
        Parameters:
            bounds_dict : dict
                A dictionary with boundary conditions as numpy (H,2) arrays with two columns for each type of hyperparameter.
            scale : float
                Scale the boundary conditions.
            log : bool
                Whether to use hyperparameters in log-scale or not.
            use_derivatives : bool
                Whether the derivatives of the target are used in the model. 
                The boundary conditions of the length-scale hyperparameter(s) will change with the use_derivatives. 
                The use_derivatives will be updated when update_bounds is called.
            use_prior_mean : bool
                Whether to use the prior mean to calculate the boundary of the prefactor hyperparameter.
                If use_prior_mean=False the minimum and maximum target differences are used as the boundary conditions. 
        """
        super().__init__(bounds_dict=bounds_dict,scale=scale,log=log,use_derivatives=use_derivatives,use_prior_mean=use_prior_mean,**kwargs)
    
    def length_bound(self,X,l_dim,**kwargs):
        " Get the minimum and maximum ranges of the length-scale in the educated guess regime within a scale. "
        # Get the minimum and maximum scaling factors
        exp_lower=0.2/self.scale
        exp_max=4.0*self.scale
        # Scale the convergence if derivatives of targets are used
        if self.use_derivatives:
            exp_lower=exp_lower*0.05
        lengths=np.zeros((l_dim,2))
        # If only one features is given then end 
        if len(X)==1:
            lengths[:,0]=exp_lower
            lengths[:,1]=exp_max
            if self.log:
                return np.log(lengths)
            return lengths
        # Ensure that the features are a matrix
        if not isinstance(X[0],(list,np.ndarray)):
            X=np.array([fp.get_vector() for fp in X])
        for d in range(l_dim):
            # Calculate distances 
            if l_dim==1:
                dis=pdist(X)
            else:
                dis=pdist(X[:,d:d+1])
            # Calculate the maximum length-scale
            dis_max=exp_max*np.median(dis)
            if dis_max==0.0:
                dis_min,dis_max=exp_lower,exp_max
            else:
                # Calculate the minimum length-scale from the nearest neighbor distance
                dis_min=exp_lower*np.median(self.nearest_neighbors(dis))
                if dis_min==0.0:
                    dis_min=exp_lower
            # Transform into log-scale if specified
            lengths[d,0],lengths[d,1]=dis_min,dis_max
        if self.log:
            return np.log(lengths)
        return lengths
    