import numpy as np
from scipy.spatial.distance import squareform
from .kernel import Kernel

class SE(Kernel):
    def __init__(self,use_derivatives=False,use_fingerprint=False,hp={},**kwargs):
        """
        The Kernel class with hyperparameters.
        Squared exponential or radial basis kernel class.
        Parameters:
            use_derivatives: bool
                Whether to use the derivatives of the targets.
            use_fingerprint: bool
                Whether fingerprint objects is given or arrays.
            hp: dict
                A dictionary of the hyperparameters in the log-space.
                The hyperparameters should be given as flatten arrays, 
                like hp=dict(length=np.array([-0.7])).
        """
        super().__init__(use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         hp=hp,
                         **kwargs)

    def get_KXX(self,features,**kwargs):
        # Scale features or fingerprints with their length-scales
        if self.use_fingerprint:
            X=self.get_arrays(features)*np.exp(-self.hp['length'][0])
        else:
            X=features*np.exp(-self.hp['length'][0])
        # Calculate the symmetric scaled distance matrix
        D=self.get_symmetric_absolute_distances(X,metric='sqeuclidean')
        # Calculate the normal covariance matrix 
        K=squareform(np.exp((-0.5)*D))
        np.fill_diagonal(K,1.0)
        # Whether to the extended covariance matrix for derivative of targets
        if self.use_derivatives:
            if self.use_fingerprint:
                return self.get_KXX_ext_fp(features,X,D,K)
            return self.get_KXX_ext(features,X,D,K)
        return K
    
    def get_KQX(self,features,features2,get_derivatives=True,**kwargs):
        # Scale features or fingerprints with their length-scales
        length_scale=np.exp(-self.hp['length'][0])
        if self.use_fingerprint:
            Q,X=self.get_arrays(features,features2)
            Q=Q*length_scale
            X=X*length_scale
        else:
            Q=features*length_scale
            X=features2*length_scale
        D=self.get_absolute_distances(Q,X,metric='sqeuclidean')
        K=np.exp((-0.5)*D)
        if get_derivatives or self.use_derivatives:
            if self.use_fingerprint:
                return self.get_KQX_ext_fp(features,features2,Q,X,D,K,get_derivatives=get_derivatives)
            return self.get_KQX_ext(features,features2,Q,X,D,K,get_derivatives=get_derivatives)
        return np.exp((-0.5)*D)
    
    def get_KXX_ext(self,features,X,D,K,**kwargs):
        """
        Make the extended symmetric kernel matrix without fingerprints.
        Parameters:
            features: (N,D) array
                Features with N data points.
            X: (N,D) array
                Features in the scaled feature space.
            D: (N*(N-1)/2) array
                All squared euclidean distances.
            K: (N,N) array
                The covariance matrix without derivatives of the features.
        Returns:
            (N*D+N,N*D+N) array : The extended symmetric kernel matrix.
        """
        # Get dimensions
        nd1,xdim=np.shape(X)
        # Get the derivative and hessian of the scaled distance matrix
        dDpre,dD=self.get_distance_derivative(X,X,nd1,nd1,xdim,axis=0)
        ddDpre=-2.0*np.exp(-2*self.hp['length'][0])
        # The first derivative of the kernel
        dKpre,dK=self.get_derivative_K(K)
        dKdD=(-dDpre*dKpre)*dK
        # The hessian of the kernel 
        ddKpre,ddK=self.get_hessian_K(K)
        ddKdD=((-dDpre*dDpre*ddKpre)*ddK)*dD
        dKddD=(ddDpre*dKpre)*dK 
        # Calculate the full symmetric kernel matrix
        Kext=np.zeros((nd1*(xdim+1),nd1*(xdim+1)))
        Kext[:nd1,:nd1]=K.copy()
        ## Derivative part
        Kext[:nd1,nd1:]=np.transpose(dKdD*dD,(1,0,2)).reshape(nd1,nd1*xdim)
        Kext[nd1:,:nd1]=Kext[:nd1,nd1:].T
        ## Hessian part
        for d1 in range(1,xdim):
            Kext[nd1*d1:nd1*(d1+1),nd1*(d1+1):]=np.transpose(ddKdD[d1:]*dD[d1-1],(1,0,2)).reshape(nd1,nd1*(xdim-d1))
            Kext[nd1*(d1+1):,nd1*d1:nd1*(d1+1)]=Kext[nd1*d1:nd1*(d1+1),nd1*(d1+1):].T   
            Kext[nd1*d1:nd1*(d1+1),nd1*d1:nd1*(d1+1)]=(ddKdD[d1-1]*dD[d1-1])+dKddD
        Kext[nd1*xdim:nd1*(xdim+1),nd1*xdim:nd1*(xdim+1)]=(ddKdD[xdim-1]*dD[xdim-1])+dKddD
        return Kext

    def get_KXX_ext_fp(self,features,X,D,K,**kwargs):
        """
        Make the extended symmetric kernel matrix with fingerprints.
        Parameters:
            features: (N) list of fingerprint objects
                Features with N data points.
            X: (N,D) array
                Features in the scaled fingerprint space.
            D: (N*(N-1)/2) array
                All squared euclidean distances.
            K: (N,N) array
                The covariance matrix without derivatives of the features.
        Returns:
            (N*Dx+N,N*Dx+N) array : The extended symmetric kernel matrix.
        """
        # Get dimensions
        nd1=len(X)
        xdim=self.get_derivative_dimension(features)
        # Get the derivative and hessian of the scaled distance matrix
        fp_deriv=self.get_fp_deriv(features)
        dDpre,dD=self.get_distance_derivative_fp(X,fp_deriv,X=None,axis=0,**kwargs)
        ddDpre,ddD=self.get_distance_hessian_fp(fp_deriv,fp_deriv)
        # The first derivative of the kernel
        dKpre,dK=self.get_derivative_K(K)
        dKdD=(-dDpre*dKpre)*dK
        # The hessian of the kernel 
        ddKpre,ddK=self.get_hessian_K(K)
        ddKdD=((dDpre*dDpre*ddKpre)*ddK)*np.transpose(dD,(0,2,1))
        dKddD=(ddDpre*dKpre)*dK 
        # Calculate the full symmetric kernel matrix
        Kext=np.zeros((nd1*(xdim+1),nd1*(xdim+1)))
        Kext[:nd1,:nd1]=K.copy()
        ## Derivative part
        Kext[:nd1,nd1:]=np.transpose(dKdD*dD,(1,0,2)).reshape(nd1,nd1*xdim)
        Kext[nd1:,:nd1]=Kext[:nd1,nd1:].T
        ## Hessian part
        for d1 in range(1,xdim):
            Kext[nd1*d1:nd1*(d1+1),nd1*d1:]=np.transpose((ddKdD[d1-1]*dD[d1-1:])+(dKddD*ddD[d1-1,d1-1:]),(1,0,2)).reshape(nd1,nd1*(xdim-d1+1))
            Kext[nd1*(d1+1):,nd1*d1:nd1*(d1+1)]=Kext[nd1*d1:nd1*(d1+1),nd1*(d1+1):].T     
        Kext[nd1*xdim:nd1*(xdim+1),nd1*xdim:nd1*(xdim+1)]=(ddKdD[xdim-1]*dD[xdim-1])+(dKddD*ddD[xdim-1,xdim-1:])
        return Kext

    def get_KQX_ext(self,features,features2,Q,X,D,K,get_derivatives=True,**kwargs):
        """
        Make the extended kernel matrix without fingerprints.
        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Features with M data points.
            features2: (N,D) array or (N) list of fingerprint objects
                Features with N data points and D dimensions. 
            Q: (M,D) array
                Features in the scaled feature space.
            X: (N,D) array
                Features in the scaled feature space.
            D: (M,N) array
                All squared euclidean distances.
            K: (M,N) array
                The covariance matrix without derivatives of the features.
            get_derivatives: bool
                Whether to predict derivatives of target.
        Returns:
            (M*D+N,N*D+N) array : The extended kernel matrix.
        """
        # Get dimensions
        nd1=len(Q)
        nd2,xdim=np.shape(X)
        nrows=nd1*(xdim+1) if get_derivatives else nd1
        ncol=nd2*(xdim+1) if self.use_derivatives else nd2
        # The full kernel matrix
        Kext=np.zeros((nrows,ncol))
        Kext[:nd1,:nd2]=K.copy()
        # Get the derivative of the scaled distance matrix
        dDpre,dD=self.get_distance_derivative(Q,X,nd1,nd2,xdim,axis=0)
        # The first derivative of the kernel
        dKpre,dK=self.get_derivative_K(K)
        dKdD=(-dDpre*dKpre)*dK
        btensor=dKdD*dD
        if self.use_derivatives:
            # Derivative part of X
            Kext[:nd1,nd2:]=np.transpose(btensor,(1,0,2)).reshape(nd1,nd2*xdim)
        if get_derivatives:
            # Derivative part of X
            Kext[nd1:,:nd2]=-(btensor).reshape(nd1*xdim,nd2)
            # Hessian part
            if self.use_derivatives:
                ddKpre,ddK=self.get_hessian_K(K)
                ddKdD=((-dDpre*dDpre*ddKpre)*ddK)*dD
                ddDpre=-2.0*np.exp(-2*self.hp['length'][0])
                dKddD=(ddDpre*dKpre)*dK
                btensor=ddKdD[:,None,:,:]*dD
                btensor[range(xdim),range(xdim),:,:]+=dKddD
                Kext[nd1:,nd2:]=np.transpose(btensor,(0,2,1,3)).reshape(nd1*xdim,nd2*xdim)
        return Kext
    
    def get_KQX_ext_fp(self,features,features2,Q,X,D,K,get_derivatives=True,**kwargs):
        """
        Make the extended kernel matrix with fingerprints.
        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Features with M data points.
            features2: (N,D) array or (N) list of fingerprint objects
                Features with N data points and D dimensions. 
            Q: (M,D) array
                Features in the scaled feature space.
            X: (N,D) array
                Features in the scaled feature space.
            D: (M,N) array
                All squared euclidean distances.
            K: (M,N) array
                The covariance matrix without derivatives of the features.
            get_derivatives: bool
                Whether to predict derivatives of target.
        Returns:
            (M*Dx+N,N*Dx+N) array : The extended kernel matrix.
        """
        # Get dimensions
        nd1,nd2=len(Q),len(X)
        xdim=self.get_derivative_dimension(features)
        nrows=nd1*(xdim+1) if get_derivatives else nd1
        ncol=nd2*(xdim+1) if self.use_derivatives else nd2
        # The full kernel matrix
        Kext=np.zeros((nrows,ncol))
        Kext[:nd1,:nd2]=K.copy()
        # The first derivative of the kernel
        dKpre,dK=self.get_derivative_K(K)
        # Get the derivative of the scaled distance matrix for X
        if self.use_derivatives:
            fp_deriv2=self.get_fp_deriv(features2)
            dDpre2,dD2=self.get_distance_derivative_fp(Q,fp_deriv2,X=X,axis=1,**kwargs)
            dKdD=(dDpre2*dKpre)*dK
            Kext[:nd1,nd2:]=np.transpose(dKdD*dD2,(1,0,2)).reshape(nd1,nd2*xdim)
        # Get the derivative of the scaled distance matrix for Q
        if get_derivatives:
            fp_deriv1=self.get_fp_deriv(features)
            dDpre1,dD1=self.get_distance_derivative_fp(Q,fp_deriv1,X=X,axis=0,**kwargs)
            if self.use_derivatives:
                ## Derivative part of Q
                Kext[nd1:,:nd2]=((-dKdD)*dD1).reshape(nd1*xdim,nd2)
                ## Hessian part
                ddDpre,ddD=self.get_distance_hessian_fp(fp_deriv1,fp_deriv2)
                ### The hessian of the kernel 
                ddKpre,ddK=self.get_hessian_K(K)
                ddKdD=((dDpre1*dDpre2*ddKpre)*ddK)*dD1
                dKddD=(ddDpre*dKpre)*dK 
                Kext[nd1:,nd2:]=np.transpose(np.einsum('ijk,ljk->iljk',ddKdD,dD2,optimize=True)+(ddD*dKddD),(0,2,1,3)).reshape(nd1*xdim,nd2*xdim)
            else:
                ## Derivative part of Q
                dKdD=(dDpre1*dKpre)*dK
                Kext[nd1:,:nd2]=(dKdD*dD1).reshape(nd1*xdim,nd2)
        return Kext

    def get_derivative_K(self,K,**kwargs):
        """
        Make the derivative of the kernel matrix wrt. the scaled distance matrix.
        The prefactors of the kernel and distance are cancelled out except for the length scale.
        The distance matrix contains one of the length scales.
        Parameters:
            K : (N,M) array
                The kernel matrix without derivatives.
        Returns:
            float: The outer derivative value.
            and
            (N,M) array: The derivative kernel matrix
        """
        return -0.5,K
    
    def get_hessian_K(self,K,**kwargs):
        """
        Make the hessian of the kernel matrix wrt. the scaled distance matrix.
        The prefactors of the kernel and distance are cancelled out except for the length scale.
        The distance matrices contain one of the length scales.
        Parameters:
            K : (N,M) array
                The kernel matrix without derivatives.
        Returns:
            float: The outer hessian value.
            and
            (N,M) array: The hessian kernel matrix
        """
        return 0.25,K
    
    def get_distance_derivative(self,Q,X,nd1,nd2,dim,axis=0,**kwargs):
        " Get the derivative of the scaled distance matrix wrt the features/fingerprint. "    
        dDpre=2.0*np.exp(-self.hp['length'][0]) if axis==0 else -2.0*np.exp(-self.hp['length'][0])
        return dDpre,Q.T.reshape(dim,nd1,1)-X.T.reshape(dim,1,nd2)
    
    def get_distance_derivative_fp(self,Q,fp_deriv,X=None,axis=0,**kwargs):
        " Get the derivative of the distance matrix wrt the features/fingerprint. "  
        dDpre=2.0*np.exp(-self.hp['length'][0]) if axis==0 else -2.0*np.exp(-self.hp['length'][0])
        if X is None:
            Q_chain=np.einsum('lj,ikj->ilk',Q,fp_deriv)
            return dDpre,Q_chain-np.diagonal(Q_chain,axis1=1,axis2=2)[:,None,:]
        if axis==0:
            Q_chain=np.einsum('kj,ikj->ik',Q,fp_deriv)
            X_chain=np.einsum('lj,ikj->ikl',X,fp_deriv)
            return dDpre,Q_chain[:,:,None]-X_chain
        Q_chain=np.einsum('lj,ikj->ilk',Q,fp_deriv)
        X_chain=np.einsum('kj,ikj->ik',X,fp_deriv)
        return dDpre,Q_chain-X_chain[:,None,:]
    
    def get_distance_hessian(self,**kwargs):
        " Get the derivative of the scaled distance matrix wrt the features/fingerprint. "    
        dDpre=-2.0*np.exp(-2*self.hp['length'][0])
        return dDpre,1.0
    
    def get_distance_hessian_fp(self,fp_deriv1,fp_deriv2,**kwargs):
        " Get the derivative of the scaled distance matrix wrt the features/fingerprint. "    
        dDpre=-2.0*np.exp(-2*self.hp['length'][0])
        return dDpre,np.einsum('dji,eki->dejk',fp_deriv1,fp_deriv2,optimize=True)

    def diag(self,features,get_derivatives=True,**kwargs):
        nd1=len(features)
        K_diag=np.ones(nd1)
        if get_derivatives:
            if self.use_fingerprint:
                fp_deriv=self.get_fp_deriv(features)
                Kdd_diag=np.einsum('dij,dij->di',fp_deriv,fp_deriv,optimize=True).reshape(-1)
                return np.append(K_diag,np.exp(-2.0*self.hp['length'][0])*Kdd_diag)
            return np.append(K_diag,np.exp(-2.0*self.hp['length'][0])*np.ones(nd1*len(features[0])))
        return K_diag
        
    def get_gradients(self,features,hp,KXX,correction=True,**kwargs):
        hp_deriv={}
        if 'length' in hp:
            if self.use_fingerprint:
                X=self.get_arrays(features)*np.exp(-self.hp['length'][0])
            else:
                X=features*np.exp(-self.hp['length'][0])
            D=squareform(self.get_symmetric_absolute_distances(X,metric='sqeuclidean'))
            if self.use_derivatives:
                nd1=len(features)
                xdim=self.get_derivative_dimension(features) if self.use_fingerprint else len(X[0])
                K=KXX[:nd1,:nd1].copy()
                K_diag=np.diag(KXX)
                Kd=KXX.copy()
                Kd[:nd1,:nd1]=Kd[:nd1,:nd1]*D
                D2=D-2
                Kd[:nd1,nd1:]=Kd[:nd1,nd1:]*np.tile(D2,(1,xdim))
                Kd[nd1:,:nd1]=Kd[:nd1,nd1:].T
                ddKpre,ddK=self.get_hessian_K(K)
                if self.use_fingerprint:
                    fp_deriv=self.get_fp_deriv(features)
                    dDpre,dD=self.get_distance_derivative_fp(X,fp_deriv,X=None,axis=0,**kwargs)
                    ddKdD=((dDpre*dDpre*ddKpre)*ddK)*np.transpose(dD,(0,2,1))
                else:
                    dDpre,dD=self.get_distance_derivative(X,X,nd1,nd1,xdim,axis=0)
                    ddKdD=((-dDpre*dDpre*ddKpre)*ddK)*dD
                for d1 in range(1,xdim):
                    ddKdDdD=2*np.transpose(ddKdD[d1-1]*dD[d1-1:],(1,0,2)).reshape(nd1,nd1*(xdim-d1+1))
                    Kd[nd1*d1:nd1*(d1+1),nd1*d1:]=(Kd[nd1*d1:nd1*(d1+1),nd1*d1:]*np.tile(D2,(1,xdim-d1+1)))-ddKdDdD
                    Kd[nd1*(d1+1):,nd1*d1:nd1*(d1+1)]=Kd[nd1*d1:nd1*(d1+1),nd1*(d1+1):].T   
                Kd[nd1*xdim:nd1*(xdim+1),nd1*xdim:nd1*(xdim+1)]=Kd[nd1*xdim:nd1*(xdim+1),nd1*xdim:nd1*(xdim+1)]*D2-(2*ddKdD[xdim-1]*dD[xdim-1])
                if correction:
                    Kd[range(nd1*xdim),range(nd1*xdim)]+=(1/(1/(2.3e-16)-(len(K_diag)**2)))*(2*np.sum(K_diag))*(-2*np.sum(K_diag[nd1:]))
            else:
                Kd=D*KXX
            hp_deriv['length']=np.array([Kd])
        return hp_deriv
