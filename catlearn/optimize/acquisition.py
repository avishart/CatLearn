import numpy as np

class Acquisition():

    def __init__(self,mode='energy',objective='min',kappa=2,unc_convergence=0.05,stationary_point_found=False):
        """
        Acquisition function class.
        Parameters:
            mode : string
                The type of acquisition function used
                Available:
                    - 'energy': Predicted energy.
                    - 'uncertainty': Predicted uncertainty.
                    - 'ucb': Predicted energy plus uncertainty.
                    - 'lcb': Predicted energy minus uncertainty.
                    - 'ue': Predicted uncertainty every second time else predicted energy.
                    - 'ume': Predicted uncertainty when it is is larger than unc_convergence else predicted energy.
                    - 'umue': Predicted uncertainty when it is is larger than unc_convergence else 'ue'.
                    - 'sume': 'ume' if stationary point is found else 'ue'.
                    - 'umucb': Predicted uncertainty when it is is larger than unc_convergence else 'ucb'.
                    - 'umlcb': Predicted uncertainty when it is is larger than unc_convergence else 'lcb'.
            objective : string
                How to sort a list of acquisition functions
                Available:
                    - 'min': Sort after the smallest values.
                    - 'max': Sort after the largest values.
                    - 'random' : Sort randomly
            kappa : int or string
                The scale of the uncertainty when 'ucb' or 'lcb' is chosen.
                If 'random' is set for kappa, the kappa value is chosen randomly between 0 and 5  
            unc_convergence : float
                The uncertainty convergence criteria.
            stationary_point_found : bool
                If the stationary point is found.
        """
        self.update(mode)
        self.objective=objective
        self.kappa=kappa
        self.unc_convergence=unc_convergence
        self.stationary_point_found=stationary_point_found
        self.iter=0
        

    def update(self,mode):
        self.mode=mode.lower()
        acq={'energy':self.ener, 'uncertainty':self.unc, 'ucb':self.ucb, 'lcb':self.lcb, \
                'ue':self.ue, 'ume':self.ume, 'umue':self.umue, 'sume':self.sume, 'umucb':self.umucb, 'umlcb':self.umlcb}
        self.calculate=acq[self.mode]
        pass

    def choose(self,candidates):
        " Sort a list of acquisition function values "
        if self.objective.lower()=='min':
            return np.argsort(candidates)
        elif self.objective.lower()=='max':
            return np.argsort(candidates)[::-1]
        elif self.objective.lower()=='random':
            return np.random.permutation(list(range(len(candidates))))
        return np.random.permutation(list(range(len(candidates))))

    def ener(self,energy,uncertainty=None):
        " Predicted energy "
        return energy

    def unc(self,energy,uncertainty=None):
        " Predicted uncertainty "
        return uncertainty

    def ucb(self,energy,uncertainty=None):
        " Predicted energy plus uncertainty "
        kappa=np.random.uniform(0,5) if self.kappa=='random' else abs(self.kappa)
        return energy+kappa*uncertainty

    def lcb(self,energy,uncertainty=None):
        " Predicted energy minus uncertainty "
        kappa=np.random.uniform(0,5) if self.kappa=='random' else abs(self.kappa)
        return energy-kappa*uncertainty

    def ue(self,energy,uncertainty=None):
        " Predicted uncertainty every second time else predicted energy "
        if self.iter%2==0:
            return uncertainty
        return energy

    def ume(self,energy,uncertainty=None):
        " Predicted uncertainty when it is is larger than unc_convergence else predicted energy "
        if np.max([uncertainty])<self.unc_convergence:
            return energy
        return uncertainty

    def umue(self,energy,uncertainty=None):
        " Predicted uncertainty when it is is larger than unc_convergence else 'ue' "
        if np.max([uncertainty])<self.unc_convergence:
            return self.ue(energy,uncertainty)
        return uncertainty

    def sume(self,energy,uncertainty=None):
        " 'ume' if stationary point is found else 'ue' "
        if self.stationary_point_found:
            return self.ume(energy,uncertainty)
        return self.ue(energy,uncertainty)

    def umucb(self,energy,uncertainty=None):
        " Predicted uncertainty when it is is larger than unc_convergence else 'ucb' "
        if np.max([uncertainty])<self.unc_convergence:
            return self.ucb(energy,uncertainty)
        return uncertainty

    def umlcb(self,energy,uncertainty=None):
        " Predicted uncertainty when it is is larger than unc_convergence else 'lcb' "
        if np.max([uncertainty])<self.unc_convergence:
            return self.lcb(energy,uncertainty)
        return uncertainty
    

    



