import numpy as np
from .factorized_likelihood import FactorizedLogLikelihood
from numpy.linalg import svd

class FactorizedLogLikelihoodSVD(FactorizedLogLikelihood):
    """ The factorized log-likelihood objective function that is used to optimize the hyperparameters. 
        The prefactor hyperparameter is determined from an analytical expression. 
        A SVD is performed to get the eigenvalues. 
        The relative-noise hyperparameter can be searched from a single SVD for each length-scale hyperparameter. 
    """
    
    def get_eig(self,model,X,Y):
        " Calculate the eigenvalues " 
        # Calculate the kernel with and without noise
        KXX,n_data=self.kxx_corr(model,X)
        # SVD
        U,D,Vt=svd(KXX,hermitian=True)
        # Subtract the prior mean to the training target
        Y_p=self.y_prior(X,Y,model,D=D,U=U)
        UTY=np.matmul(Vt,Y_p).reshape(-1)**2
        return D,U,Y_p,UTY,KXX,n_data
      