import numpy as np
from .factorized_likelihood import FactorizedLogLikelihood
from numpy.linalg import svd

class FactorizedLogLikelihoodSVD(FactorizedLogLikelihood):
    
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
    