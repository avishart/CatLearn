from .method import OptimizerMethod
import ase
from ase.parallel import world
from ase.optimize import FIRE
from numpy import isnan


class LocalOptimizer(OptimizerMethod):
    def __init__(
        self,
        atoms,
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        """
        The LocalOptimizer is used to run a local optimization on
        a given structure.
        The LocalOptimizer is applicable to be used with active learning.

        Parameters:
            atoms: Atoms instance
                The instance to be optimized.
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the parameters
        self.update_arguments(
            atoms=atoms,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def run(
        self,
        fmax=0.05,
        steps=1000,
        max_unc=None,
        dtrust=None,
        unc_convergence=None,
        **kwargs,
    ):
        # Check if the optimization can take any steps
        if steps <= 0:
            return self._converged
        # Run the local optimization
        with self.local_opt(
            self.optimizable, **self.local_opt_kwargs
        ) as optimizer:
            if max_unc is None and dtrust is None:
                optimizer.run(fmax=fmax, steps=steps)
                converged = optimizer.converged()
            else:
                converged = self.run_max_unc(
                    optimizer=optimizer,
                    fmax=fmax,
                    steps=steps,
                    max_unc=max_unc,
                    dtrust=dtrust,
                    **kwargs,
                )
            # Check if the optimization is converged
            self._converged = self.check_convergence(
                converged=converged,
                max_unc=max_unc,
                dtrust=dtrust,
                unc_convergence=unc_convergence,
            )
        # Return whether the optimization is converged
        return self._converged

    def run_max_unc(
        self,
        optimizer,
        fmax=0.05,
        steps=1000,
        max_unc=None,
        dtrust=None,
        **kwargs,
    ):
        """
        Run the optimization with a maximum uncertainty.

        Parameters:
            optimizer: ASE optimizer object
                The optimizer object.
            fmax: float
                The maximum force allowed on an atom.
            steps: int
                The maximum number of steps allowed.
            max_unc: float
                The maximum uncertainty allowed on a structure.
            dtrust: float
                The distance trust criterion.

        Returns:
            converged: bool
                Whether the optimization is converged.
        """
        # Set the converged parameter
        converged = False
        # Make a copy of the atoms
        while self.steps < steps:
            # Check if the maximum number of steps is reached
            if self.steps >= steps:
                self.message("The maximum number of steps is reached.")
                break
            # Run a local optimization step
            _converged = self.run_max_unc_step(optimizer, fmax=fmax, **kwargs)
            # Check if the uncertainty is above the maximum allowed
            if max_unc is not None:
                # Get the uncertainty of the atoms
                unc = self.get_uncertainty()
                if unc > max_unc:
                    self.message("Uncertainty is above the maximum allowed.")
                    break
            # Check if the structures are within the trust distance
            if dtrust is not None:
                within_dtrust = self.is_within_dtrust(dtrust=dtrust)
                if not within_dtrust:
                    self.message("Outside of the trust distance.")
                    break
            # Check if there is a problem with the calculation
            energy = self.get_potential_energy()
            if isnan(energy):
                self.message("The energy is NaN.")
                break
            # Check if the optimization is converged
            if _converged:
                converged = True
                break
        return converged

    def setup_local_optimizer(self, local_opt=None, local_opt_kwargs={}):
        """
        Setup the local optimizer.

        Parameters:
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
        """
        self.local_opt_kwargs = dict()
        if not self.verbose:
            self.local_opt_kwargs["logfile"] = None
        if local_opt is None:
            local_opt = FIRE
            self.local_opt_kwargs.update(
                dict(dt=0.05, maxstep=0.2, a=1.0, astart=1.0, fa=0.999)
            )
        self.local_opt = local_opt
        self.local_opt_kwargs.update(local_opt_kwargs)
        return self

    def is_energy_minimized(self):
        return True

    def is_parallel_allowed(self):
        return False

    def update_arguments(
        self,
        atoms=None,
        local_opt=None,
        local_opt_kwargs={},
        parallel_run=None,
        comm=None,
        verbose=None,
        seed=None,
        **kwargs,
    ):
        """
        Update the instance with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            atoms: Atoms instance
                The instance to be optimized.
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the parameters in the parent class
        super().update_arguments(
            optimizable=atoms,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        # Set the local optimizer
        if local_opt is not None and local_opt_kwargs is not None:
            self.setup_local_optimizer(local_opt, local_opt_kwargs)
        elif local_opt is not None:
            self.setup_local_optimizer(self.local_opt)
        elif local_opt_kwargs is not None:
            self.setup_local_optimizer(self.local_opt, local_opt_kwargs)
        return self

    def run_max_unc_step(self, optimizer, fmax=0.05, **kwargs):
        """
        Run a local optimization step.
        The ASE optimizer is dependent on the ASE version.
        """
        if ase.__version__ >= "3.23":
            optimizer.run(fmax=fmax, steps=1, **kwargs)
        else:
            optimizer.run(fmax=fmax, steps=self.steps + 1, **kwargs)
        self.steps += 1
        return optimizer.converged()

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            atoms=self.optimizable,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
            seed=self.seed,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
