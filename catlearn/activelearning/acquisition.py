from numpy import argsort, max as max_
from numpy.random import default_rng, Generator, RandomState
from scipy.stats import norm


class Acquisition:
    def __init__(self, objective="min", seed=None, **kwargs):
        """
        Acquisition function class.

        Parameters:
            objective : string
                How to sort a list of acquisition functions
                Available:
                    - 'min': Sort after the smallest values.
                    - 'max': Sort after the largest values.
                    - 'random' : Sort randomly
            seed: int (optional)
                The random seed.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        self.update_arguments(objective=objective, seed=seed, **kwargs)

    def calculate(self, energy, uncertainty=None, **kwargs):
        "Calculate the acqusition function value."
        raise NotImplementedError()

    def choose(self, candidates):
        "Sort a list of acquisition function values."
        if self.objective == "min":
            return argsort(candidates)
        elif self.objective == "max":
            return argsort(candidates)[::-1]
        return self.rng.permutation(list(range(len(candidates))))

    def objective_value(self, value):
        "Return the objective value."
        if self.objective == "min":
            return -value
        return value

    def update_arguments(self, objective=None, seed=None, **kwargs):
        "Set the parameters of the Acquisition function class."
        # Set the seed
        if seed is not None or not hasattr(self, "seed"):
            self.set_seed(seed)
        # Set the objective
        if objective is not None:
            self.objective = objective.lower()
        return self

    def set_seed(self, seed=None):
        "Set the random seed."
        if seed is not None:
            self.seed = seed
            if isinstance(seed, int):
                self.rng = default_rng(self.seed)
            elif isinstance(seed, Generator) or isinstance(seed, RandomState):
                self.rng = seed
        else:
            self.seed = None
            self.rng = default_rng()
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(objective=self.objective, seed=self.seed)
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)


class AcqEnergy(Acquisition):
    def __init__(self, objective="min", seed=None, **kwargs):
        "The predicted energy as the acqusition function."
        super().__init__(objective=objective, seed=seed, **kwargs)

    def calculate(self, energy, uncertainty=None, **kwargs):
        "Calculate the acqusition function value as the predicted energy."
        return energy


class AcqUncertainty(Acquisition):
    def __init__(self, objective="min", seed=None, **kwargs):
        "The predicted uncertainty as the acqusition function."
        super().__init__(objective=objective, seed=seed, **kwargs)

    def calculate(self, energy, uncertainty=None, **kwargs):
        "Calculate the acqusition function value as the predicted uncertainty."
        return uncertainty


class AcqUCB(Acquisition):
    def __init__(
        self,
        objective="max",
        seed=None,
        kappa=2.0,
        kappamax=3.0,
        **kwargs,
    ):
        """
        The predicted upper confidence interval (ucb) as
        the acqusition function.
        """
        self.update_arguments(
            objective=objective,
            seed=seed,
            kappa=kappa,
            kappamax=kappamax,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        "Calculate the acqusition function value as the predicted ucb."
        kappa = self.get_kappa()
        return energy + kappa * uncertainty

    def get_kappa(self):
        "Get the kappa value."
        if isinstance(self.kappa, str):
            return self.rng.uniform(0, self.kappamax)
        return self.kappa

    def update_arguments(
        self,
        objective=None,
        seed=None,
        kappa=None,
        kappamax=None,
        **kwargs,
    ):
        "Set the parameters of the Acquisition function class."
        # Set the parameters in the parent class
        super().update_arguments(
            objective=objective,
            seed=seed,
            **kwargs,
        )
        # Set the kappa value
        if kappa is not None:
            if isinstance(kappa, (float, int)):
                kappa = abs(kappa)
            self.kappa = kappa
        # Set the kappamax value
        if kappamax is not None:
            self.kappamax = abs(kappamax)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            objective=self.objective,
            seed=self.seed,
            kappa=self.kappa,
            kappamax=self.kappamax,
            seed=self.seed,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AcqLCB(AcqUCB):
    def __init__(
        self,
        objective="min",
        seed=None,
        kappa=2.0,
        kappamax=3.0,
        **kwargs,
    ):
        """
        The predicted lower confidence interval (lcb) as
        the acqusition function.
        """
        super().__init__(
            objective=objective,
            seed=seed,
            kappa=kappa,
            kappamax=kappamax,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        "Calculate the acqusition function value as the predicted ucb."
        kappa = self.get_kappa()
        return energy - kappa * uncertainty


class AcqIter(Acquisition):
    def __init__(self, objective="max", seed=None, niter=2, **kwargs):
        """
        The predicted energy or uncertainty dependent on
        the iteration as the acqusition function.
        """
        super().__init__(objective=objective, seed=seed, niter=niter, **kwargs)
        self.iter = 0

    def calculate(self, energy, uncertainty=None, **kwargs):
        """
        Calculate the acqusition function value as
        the predicted energy or uncertainty.
        """
        self.iter += 1
        if (self.iter) % self.niter == 0:
            return energy
        return uncertainty

    def update_arguments(
        self,
        objective=None,
        seed=None,
        niter=None,
        **kwargs,
    ):
        "Set the parameters of the Acquisition function class."
        # Set the parameters in the parent class
        super().update_arguments(
            objective=objective,
            seed=seed,
            **kwargs,
        )
        # Set the number of iterations
        if niter is not None:
            self.niter = abs(niter)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            objective=self.objective,
            seed=self.seed,
            niter=self.niter,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AcqUME(Acquisition):
    def __init__(
        self, objective="max", seed=None, unc_convergence=0.05, **kwargs
    ):
        """
        The predicted uncertainty when it is larger than unc_convergence
        else predicted energy as the acqusition function.
        """
        super().__init__(
            objective=objective,
            seed=seed,
            unc_convergence=unc_convergence,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        """
        Calculate the acqusition function value as the predicted uncertainty
        when it is is larger than unc_convergence else predicted energy.
        """
        if max_([uncertainty]) < self.unc_convergence:
            return energy
        return self.objective_value(uncertainty)

    def update_arguments(self, objective=None, unc_convergence=None, **kwargs):
        "Set the parameters of the Acquisition function class."
        # Set the parameters in the parent class
        super().update_arguments(
            objective=objective,
            seed=None,
            **kwargs,
        )
        # Set the unc_convergence value
        if unc_convergence is not None:
            self.unc_convergence = abs(unc_convergence)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            objective=self.objective,
            seed=self.seed,
            unc_convergence=self.unc_convergence,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AcqUUCB(AcqUCB):
    def __init__(
        self,
        objective="max",
        seed=None,
        kappa=2.0,
        kappamax=3.0,
        unc_convergence=0.05,
        **kwargs,
    ):
        """
        The predicted uncertainty when it is larger than unc_convergence
        else upper confidence interval (ucb) as the acqusition function.
        """
        self.update_arguments(
            objective=objective,
            seed=seed,
            kappa=kappa,
            kappamax=kappamax,
            unc_convergence=unc_convergence,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        """
        Calculate the acqusition function value as the predicted uncertainty
        when it is is larger than unc_convergence else ucb.
        """
        if max_([uncertainty]) < self.unc_convergence:
            kappa = self.get_kappa()
            return energy + kappa * uncertainty
        return uncertainty

    def update_arguments(
        self,
        objective=None,
        seed=None,
        kappa=None,
        kappamax=None,
        unc_convergence=None,
        **kwargs,
    ):
        "Set the parameters of the Acquisition function class."
        # Set the parameters in the parent class
        super().update_arguments(
            objective=objective,
            seed=seed,
            **kwargs,
        )
        # Set the kappa value
        if kappa is not None:
            if isinstance(kappa, (float, int)):
                kappa = abs(kappa)
            self.kappa = kappa
        # Set the kappamax value
        if kappamax is not None:
            self.kappamax = abs(kappamax)
        # Set the unc_convergence value
        if unc_convergence is not None:
            self.unc_convergence = abs(unc_convergence)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            objective=self.objective,
            seed=self.seed,
            kappa=self.kappa,
            kappamax=self.kappamax,
            unc_convergence=self.unc_convergence,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AcqULCB(AcqUUCB):
    def __init__(
        self,
        objective="min",
        seed=None,
        kappa=2.0,
        kappamax=3.0,
        unc_convergence=0.05,
        **kwargs,
    ):
        """
        The predicted uncertainty when it is larger than unc_convergence
        else lower confidence interval (lcb) as the acqusition function.
        """
        self.update_arguments(
            objective=objective,
            seed=seed,
            kappa=kappa,
            kappamax=kappamax,
            unc_convergence=unc_convergence,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        """
        Calculate the acqusition function value as the predicted uncertainty
        when it is is larger than unc_convergence else lcb.
        """
        if max_([uncertainty]) < self.unc_convergence:
            kappa = self.get_kappa()
            return energy - kappa * uncertainty
        return -uncertainty


class AcqEI(Acquisition):
    def __init__(self, objective="max", seed=None, ebest=None, **kwargs):
        """
        The predicted expected improvement as the acqusition function.
        """
        self.update_arguments(
            objective=objective,
            seed=seed,
            ebest=ebest,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        """
        Calculate the acqusition function value as
        the predicted expected improvement.
        """
        z = (energy - self.ebest) / uncertainty
        a = (energy - self.ebest) * norm.cdf(z) + uncertainty * norm.pdf(z)
        return self.objective_value(a)

    def update_arguments(
        self,
        objective=None,
        seed=None,
        ebest=None,
        **kwargs,
    ):
        "Set the parameters of the Acquisition function class."
        # Set the parameters in the parent class
        super().update_arguments(
            objective=objective,
            seed=seed,
            **kwargs,
        )
        # Set the ebest value
        if ebest is not None or not hasattr(self, "ebest"):
            self.ebest = ebest
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            objective=self.objective,
            seed=self.seed,
            ebest=self.ebest,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs


class AcqPI(AcqEI):
    def __init__(self, objective="max", seed=None, ebest=None, **kwargs):
        """
        The predicted probability of improvement as the acqusition function.
        """
        self.update_arguments(
            objective=objective,
            seed=seed,
            ebest=ebest,
            **kwargs,
        )

    def calculate(self, energy, uncertainty=None, **kwargs):
        """
        Calculate the acqusition function value as
        the predicted expected improvement.
        """
        z = (energy - self.ebest) / uncertainty
        return self.objective_value(norm.cdf(z))
