import numpy as np
from scipy.linalg import lstsq

class Prior:
    def __init__(self,yp=0):
        "The prior uses a baseline of the target values if given else it is at 0"
        self.yp=yp
    
    def get(self,X):
        "Give the baseline value of the target"
        return self.yp
    
    def update(self,X,Y=0):
        "The prior will use a fixed value"
        self.yp=0
        return self.yp  

    def __repr__(self):
        return "Prior(Mean={:.4f})".format(self.yp)

class Prior_zero(Prior):
    def __init__(self,yp=0):
        "The prior uses a baseline of the target values if given else it is at 0"
        Prior.__init__(self)
    
    def update(self,X,Y):
        "The prior will use the mean of the target values"
        self.yp=0
        return self.yp    

class Prior_mean(Prior):
    def __init__(self,yp=0):
        "The prior uses a baseline of the target values if given else it is at 0"
        Prior.__init__(self)
    
    def update(self,X,Y):
        "The prior will use the mean of the target values"
        self.yp=np.mean(Y)
        return self.yp

class Prior_max(Prior):
    def __init__(self,yp=0):
        "The prior uses a baseline of the target values if given else it is at 0"
        Prior.__init__(self)
    
    def update(self,X,Y):
        "The prior will use the mean of the target values"
        self.yp=np.max(Y)
        return self.yp

class Prior_min(Prior):
    def __init__(self,yp=0):
        "The prior uses a baseline of the target values if given else it is at 0"
        Prior.__init__(self)
    
    def update(self,X,Y):
        "The prior will use the mean of the target values"
        self.yp=np.min(Y)
        return self.yp
    
class Prior_regression(Prior):
    def __init__(self,yp=0):
        "The prior uses a linear regression of the target values"
        Prior.__init__(self)
    
    def get(self,X):
        "Give the baseline value of the target"
        X_new=np.append(X,np.array([1]*len(X)).reshape(-1,1),axis=1)
        return np.matmul(X_new,self.c)
    
    def update(self,X,Y=0):
        "The prior will use a linear fit"
        X_new=np.append(X,np.array([1]*len(X)).reshape(-1,1),axis=1)
        self.c=lstsq(X_new,Y.reshape(-1,1))[0]
        self.yp=np.matmul(X_new,self.c)
        return self.yp 


