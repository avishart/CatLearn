import numpy as np

class Acquisition:
    def __init__(self,objective='min',**kwargs):
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
        self.set_parameters(objective=objective,**kwargs)

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
    
    def set_parameters(self,objective=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective)
    
    def __repr__(self):
        return "{}(objective={})".format(self.__class__,self.objective)


class AcqEnergy(Acquisition):
    def __init__(self,objective='min',**kwargs):
        " The predicted energy as the acqusition function. "
        super().__init__(objective)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted energy. "
        return energy
    

class AcqUncertainty(Acquisition):
    def __init__(self,objective='min',**kwargs):
        " The predicted uncertainty as the acqusition function. "
        super().__init__(objective)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty. "
        return uncertainty
    

class AcqUCB(Acquisition):
    def __init__(self,objective='max',kappa=2.0,kappamax=3.0,**kwargs):
        " The predicted upper confidence interval (ucb) as the acqusition function. "
        self.set_parameters(objective=objective,kappa=kappa,kappamax=kappamax,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted ucb. "
        kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else self.kappa
        return energy+kappa*uncertainty
    
    def set_parameters(self,objective=None,kappa=None,kappamax=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if kappa is not None:
            if isinstance(kappa,(float,int)):
                kappa=abs(kappa)
            self.kappa=kappa
        if kappamax is not None:
            self.kappamax=abs(kappamax)
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,kappa=self.kappa,kappamax=self.kappamax)
    
    def __repr__(self):
        return "{}(objective={},kappa={},kappamax={})".format(self.__class__,self.objective,self.kappa,self.kappamax)


class AcqLCB(Acquisition):
    def __init__(self,objective='min',kappa=2.0,kappamax=3.0,**kwargs):
        " The predicted lower confidence interval (lcb) as the acqusition function. "
        self.set_parameters(objective=objective,kappa=kappa,kappamax=kappamax,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted lcb. "
        kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else self.kappa
        return energy-kappa*uncertainty
    
    def set_parameters(self,objective=None,kappa=None,kappamax=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if kappa is not None:
            if isinstance(kappa,(float,int)):
                kappa=abs(kappa)
            self.kappa=kappa
        if kappamax is not None:
            self.kappamax=abs(kappamax)
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,kappa=self.kappa,kappamax=self.kappamax)
    
    def __repr__(self):
        return "{}(objective={},kappa={},kappamax={})".format(self.__class__,self.objective,self.kappa,self.kappamax)


class AcqIter(Acquisition):
    def __init__(self,objective='max',niter=2,**kwargs):
        " The predicted energy or uncertainty dependent on the iteration as the acqusition function. "
        self.set_parameters(objective=objective,niter=niter,**kwargs)
        self.iter=0

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted energy or uncertainty. "
        self.iter+=1
        if (self.iter)%self.niter==0:
            return energy
        return uncertainty
    
    def set_parameters(self,objective=None,niter=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if niter is not None:
            self.niter=abs(niter)
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,niter=self.niter)
    
    def __repr__(self):
        return "{}(objective={},niter={})".format(self.__class__,self.objective,self.niter)
    

class AcqUME(Acquisition):
    def __init__(self,objective='max',unc_convergence=0.05,**kwargs):
        " The predicted uncertainty when it is larger than unc_convergence else predicted energy as the acqusition function. "
        self.set_parameters(objective=objective,unc_convergence=unc_convergence,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty when it is is larger than unc_convergence else predicted energy. "
        if np.max([uncertainty])<self.unc_convergence:
             return energy
        if self.objective=='max':
            return uncertainty
        return -uncertainty
    
    def set_parameters(self,objective=None,unc_convergence=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if unc_convergence is not None:
            self.unc_convergence=abs(unc_convergence)
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,unc_convergence=self.unc_convergence)
    
    def __repr__(self):
        return "{}(objective={},unc_convergence={})".format(self.__class__,self.objective,self.unc_convergence)
    

class AcqUUCB(Acquisition):
    def __init__(self,objective='max',kappa=2.0,kappamax=3.0,unc_convergence=0.05,**kwargs):
        " The predicted uncertainty when it is larger than unc_convergence else upper confidence interval (ucb) as the acqusition function. "
        self.set_parameters(objective=objective,kappa=kappa,kappamax=kappamax,unc_convergence=unc_convergence,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty when it is is larger than unc_convergence else ucb. "
        if np.max([uncertainty])<self.unc_convergence:
            kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else self.kappa
            return energy+kappa*uncertainty  
        return uncertainty
    
    def set_parameters(self,objective=None,kappa=None,kappamax=None,unc_convergence=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if kappa is not None:
            if isinstance(kappa,(float,int)):
                kappa=abs(kappa)
            self.kappa=kappa
        if kappamax is not None:
            self.kappamax=abs(kappamax)
        if unc_convergence is not None:
            self.unc_convergence=abs(unc_convergence)
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,kappa=self.kappa,kappamax=self.kappamax,unc_convergence=self.unc_convergence)
    
    def __repr__(self):
        return "{}(objective={},kappa={},kappamax={},unc_convergence={})".format(self.__class__,self.objective,self.kappa,self.kappamax,self.unc_convergence)
    

class AcqULCB(Acquisition):
    def __init__(self,objective='min',kappa=2.0,kappamax=3.0,unc_convergence=0.05,**kwargs):
        " The predicted uncertainty when it is larger than unc_convergence else lower confidence interval (lcb) as the acqusition function. "
        self.set_parameters(objective=objective,kappa=kappa,kappamax=kappamax,unc_convergence=unc_convergence,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted uncertainty when it is is larger than unc_convergence else lcb. "
        if np.max([uncertainty])<self.unc_convergence:
            kappa=np.random.uniform(0,self.kappamax) if self.kappa=='random' else abs(self.kappa)
            return energy-kappa*uncertainty  
        return -uncertainty
    
    def set_parameters(self,objective=None,kappa=None,kappamax=None,unc_convergence=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if kappa is not None:
            if isinstance(kappa,(float,int)):
                kappa=abs(kappa)
            self.kappa=kappa
        if kappamax is not None:
            self.kappamax=abs(kappamax)
        if unc_convergence is not None:
            self.unc_convergence=abs(unc_convergence)
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,kappa=self.kappa,kappamax=self.kappamax,unc_convergence=self.unc_convergence)
    
    def __repr__(self):
        return "{}(objective={},kappa={},kappamax={},unc_convergence={})".format(self.__class__,self.objective,self.kappa,self.kappamax,self.unc_convergence)
    
    
class AcqEI(Acquisition):
    def __init__(self,objective='max',ebest=None,**kwargs):
        " The predicted expected improvement as the acqusition function. "
        self.set_parameters(objective=objective,ebest=ebest,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted expected improvement. "
        from scipy.stats import norm
        z=(energy-self.ebest)/uncertainty
        a=(energy-self.ebest)*norm.cdf(z)+uncertainty*norm.pdf(z)
        if self.objective=='min':
            return -a
        return a
    
    def set_parameters(self,objective=None,ebest=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if ebest is not None:
            self.ebest=ebest
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,ebest=self.ebest)
    
    def __repr__(self):
        return "{}(objective={},ebest={})".format(self.__class__,self.objective,self.ebest)
    

class AcqPI(Acquisition):
    def __init__(self,objective='max',ebest=None,**kwargs):
        " The predicted probability of improvement as the acqusition function. "
        self.set_parameters(objective=objective,ebest=ebest,**kwargs)

    def calculate(self,energy,uncertainty=None,**kwargs):
        " Calculate the acqusition function value as the predicted expected improvement. "
        from scipy.stats import norm
        z=(energy-self.ebest)/uncertainty
        if self.objective=='min':
            return -norm.cdf(z)
        return norm.cdf(z)
    
    def set_parameters(self,objective=None,ebest=None,**kwargs):
        " Set the parameters of the Acquisition function class."
        if objective is not None:
            self.objective=objective.lower()
        if ebest is not None:
            self.ebest=ebest
        return self
    
    def copy(self):
        " Copy the Acquisition object. "
        return self.__class__(objective=self.objective,ebest=self.ebest)
    
    def __repr__(self):
        return "{}(objective={},ebest={})".format(self.__class__,self.objective,self.ebest)
