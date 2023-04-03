import numpy as np
from scipy.stats import norm

class Acquisition:
    def __init__(self,objective='min'):
        """
        Acquisition function class.
        Parameters:
            objective : string
                How to sort a list of acquisition functions
                Available:
                    - 'min': Sort after the smallest values.
                    - 'max': Sort after the largest values.
                    - 'random' : Sort randomly
        """
        self.objective=objective.lower()

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value. "
        raise NotImplementedError()

    def choose(self,candidates):
        " Sort a list of acquisition function values "
        if self.objective=='min':
            return np.argsort(candidates)
        elif self.objective=='max':
            return np.argsort(candidates)[::-1]
        elif self.objective=='random':
            return np.random.permutation(list(range(len(candidates))))
        return np.random.permutation(list(range(len(candidates))))

class AcqEnergy(Acquisition):
    def __init__(self,objective='min'):
        " The predicted energy as the acqusition function. "
        super().__init__(objective)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted energy. "
        return energy
    
class AcqUncertainty(Acquisition):
    def __init__(self,objective='min'):
        " The predicted uncertainty as the acqusition function. "
        super().__init__(objective)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty. "
        return uncertainty
    
class AcqUCB(Acquisition):
    def __init__(self,objective='max',kappa=2.0,kappamax=3):
        " The predicted upper confidence interval (ucb) as the acqusition function. "
        super().__init__(objective)
        self.kappa=kappa
        self.kappamax=kappamax

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted ucb. "
        kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else abs(self.kappa)
        return energy+kappa*uncertainty  

class AcqLCB(Acquisition):
    def __init__(self,objective='min',kappa=2.0,kappamax=3):
        " The predicted lower confidence interval (lcb) as the acqusition function. "
        super().__init__(objective)
        self.kappa=kappa
        self.kappamax=kappamax

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted lcb. "
        kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else abs(self.kappa)
        return energy-kappa*uncertainty   

class AcqIter(Acquisition):
    def __init__(self,objective='max',niter=2):
        " The predicted energy or uncertainty dependent on the iteration as the acqusition function. "
        super().__init__(objective)
        self.iter=0
        self.niter=niter

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted energy or uncertainty. "
        self.iter+=1
        if (self.iter)%self.niter==0:
            return energy
        return uncertainty
    
class AcqUME(Acquisition):
    def __init__(self,objective='max',unc_convergence=0.05):
        " The predicted uncertainty when it is larger than unc_convergence else predicted energy as the acqusition function. "
        super().__init__(objective)
        self.unc_convergence=unc_convergence

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty when it is is larger than unc_convergence else predicted energy. "
        if np.max([uncertainty])<self.unc_convergence:
             return energy
        if self.objective=='max':
            return uncertainty
        return -uncertainty
    
class AcqUUCB(Acquisition):
    def __init__(self,objective='max',kappa=2.0,kappamax=3,unc_convergence=0.05):
        " The predicted uncertainty when it is larger than unc_convergence else upper confidence interval (ucb) as the acqusition function. "
        super().__init__(objective)
        self.kappa=kappa
        self.kappamax=kappamax
        self.unc_convergence=unc_convergence

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty when it is is larger than unc_convergence else ucb. "
        if np.max([uncertainty])<self.unc_convergence:
            kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else abs(self.kappa)
            return energy+kappa*uncertainty  
        return uncertainty
    
class AcqULCB(Acquisition):
    def __init__(self,objective='min',kappa=2.0,kappamax=3,unc_convergence=0.05):
        " The predicted uncertainty when it is larger than unc_convergence else lower confidence interval (lcb) as the acqusition function. "
        super().__init__(objective)
        self.kappa=kappa
        self.kappamax=kappamax
        self.unc_convergence=unc_convergence

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty when it is is larger than unc_convergence else lcb. "
        if np.max([uncertainty])<self.unc_convergence:
            kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else abs(self.kappa)
            return energy-kappa*uncertainty  
        return -uncertainty
    
class AcqEI(Acquisition):
    def __init__(self,objective='max',ebest=None):
        " The predicted expected improvement as the acqusition function. "
        super().__init__(objective)
        self.ebest=ebest

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted expected improvement. "
        z=(energy-self.ebest)/uncertainty
        a=(energy-self.ebest)*norm.cdf(z)+uncertainty*norm.pdf(z)
        if self.objective=='min':
            return -a
        return a

class AcqPI(Acquisition):
    def __init__(self,objective='max',ebest=None):
        " The predicted probability of improvement as the acqusition function. "
        super().__init__(objective)
        self.ebest=ebest

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted expected improvement. "
        z=(energy-self.ebest)/uncertainty
        if self.objective=='min':
            return -norm.cdf(z)
        return norm.cdf(z)
