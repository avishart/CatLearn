from numpy import (
    asarray,
    max as max_,
    mean as mean_,
    nan,
    nanmax,
    ndarray,
    sqrt,
)
from numpy.linalg import norm
from numpy.random import default_rng, Generator, RandomState
from ase.io import read
from ase.parallel import world, broadcast
from ase.io.trajectory import TrajectoryWriter
import datetime
from time import time
import warnings
from ..regression.gp.calculator import BOCalculator, compare_atoms, copy_atoms
from ..regression.gp.means.max import Prior_max
from ..regression.gp.baseline import BornRepulsionCalculator


class ActiveLearning:
    """
    An active learner that is used for accelerating quantum mechanincal
    simulation methods with an active learning approach.
    """

    def __init__(
        self,
        method,
        ase_calc,
        mlcalc=None,
        acq=None,
        is_minimization=True,
        save_memory=False,
        parallel_run=False,
        copy_calc=False,
        verbose=True,
        apply_constraint=True,
        force_consistent=False,
        scale_fmax=0.8,
        use_fmax_convergence=True,
        unc_convergence=0.02,
        use_method_unc_conv=True,
        use_restart=True,
        check_unc=True,
        check_energy=True,
        check_fmax=True,
        max_unc_restart=0.05,
        n_evaluations_each=1,
        min_data=3,
        use_database_check=True,
        data_perturb=0.001,
        data_tol=1e-8,
        save_properties_traj=True,
        to_save_mlcalc=False,
        save_mlcalc_kwargs={},
        default_mlcalc_kwargs={},
        trajectory="predicted.traj",
        trainingset="evaluated.traj",
        pred_evaluated="predicted_evaluated.traj",
        converged_trajectory="converged.traj",
        initial_traj="initial_struc.traj",
        tabletxt="ml_summary.txt",
        timetxt="ml_time.txt",
        prev_calculations=None,
        restart=False,
        seed=1,
        dtype=float,
        comm=world,
        **kwargs,
    ):
        """
        Initialize the ActiveLearning instance.

        Parameters:
            method: OptimizationMethod instance
                The quantum mechanincal simulation method instance.
            ase_calc: ASE calculator instance
                ASE calculator as implemented in ASE.
            mlcalc: ML-calculator instance
                The ML-calculator instance used as surrogate surface.
                The default BOCalculator instance is used if mlcalc is None.
            acq: Acquisition class instance
                The Acquisition instance used for calculating the
                acq. function and choose a candidate to calculate next.
                The default AcqUME instance is used if acq is None.
            is_minimization: bool
                Whether it is a minimization that is performed.
                Alternative is a maximization.
            save_memory: bool
                Whether to only train the ML calculator and store all objects
                on one CPU.
                If save_memory==True then parallel optimization of
                the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.
            parallel_run: bool
                Whether to run method in parallel on multiple CPUs (True) or
                in sequence on 1 CPU (False).
            copy_calc: bool
                Whether to copy the calculator for each candidate
                in the method.
            verbose: bool
                Whether to print on screen the full output (True) or
                not (False).
            apply_constraint: bool
                Whether to apply the constrains of the ASE Atoms instance
                to the calculated forces.
                By default (apply_constraint=True) forces are 0 for
                constrained atoms and directions.
            force_consistent: bool or None.
                Use force-consistent energy calls (as opposed to the energy
                extrapolated to 0 K).
                By default force_consistent=False.
            scale_fmax: float
                The scaling of the fmax for the ML-NEB runs.
                It makes the path converge tighter on surrogate surface.
            use_fmax_convergence: bool
                Whether to use the maximum force as an convergence criterion.
            unc_convergence: float
                Maximum uncertainty for convergence in the active learning
                (in eV).
            use_method_unc_conv: bool
                Whether to use the unc_convergence as a convergence criterion
                in the optimization method.
            use_restart: bool
                Use the result from last robust iteration.
            check_unc: bool
                Check if the uncertainty is large for the restarted result and
                if it is then use the previous initial.
            check_energy: bool
                Check if the energy is larger for the restarted result than
                the previous.
            check_fmax: bool
                Check if the maximum force is larger for the restarted result
                than the initial interpolation and if so then replace it.
            max_unc_restart: float (optional)
                Maximum uncertainty (in eV) for using the structure(s) as
                the restart in the optimization method.
                If max_unc_restart is None, then the optimization is performed
                without the maximum uncertainty.
            n_evaluations_each: int
                Number of evaluations for each iteration.
            min_data: int
                The minimum number of data points in the training set before
                the active learning can converge.
            use_database_check: bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled.
                Please be aware that the predicted structure will differ from
                the structure in the database if the rattling is applied.
            data_perturb: float
                The perturbation of the data structure if it is in the database
                and use_database_check is True.
                data_perturb is the standard deviation of the normal
                distribution used to rattle the structure.
            data_tol: float
                The tolerance for the data structure if it is in the database
                and use_database_check is True.
            save_properties_traj: bool
                Whether to save the calculated properties to the trajectory.
            to_save_mlcalc: bool
                Whether to save the ML calculator to a file after training.
            save_mlcalc_kwargs: dict
                Arguments for saving the ML calculator, like the filename.
            default_mlcalc_kwargs: dict
                The default keyword arguments for the ML calculator.
            trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the predicted data.
                Or the TrajectoryWriter instance to store the predicted data.
            trainingset: str or TrajectoryWriter instance
                Trajectory filename to store the evaluated training data.
                Or the TrajectoryWriter instance to store the evaluated
                training data.
            pred_evaluated: str or TrajectoryWriter instance (optional)
                Trajectory filename to store the evaluated training data
                with predicted properties.
                Or the TrajectoryWriter instance to store the evaluated
                training data with predicted properties.
                If pred_evaluated is None, then the predicted data is
                not saved.
            converged_trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the converged structure(s).
                Or the TrajectoryWriter instance to store the converged
                structure(s).
            initial_traj: str or TrajectoryWriter instance
                Trajectory filename to store the initial structure(s).
                Or the TrajectoryWriter instance to store the initial
                structure(s).
            tabletxt: str (optional)
                Name of the .txt file where the summary table is printed.
                It is not saved to the file if tabletxt=None.
            timetxt: str (optional)
                Name of the .txt file where the time table is printed.
                It is not saved to the file if timetxt=None.
            prev_calculations: Atoms list or ASE Trajectory file.
                The user can feed previously calculated data
                for the same hypersurface.
                The previous calculations must be fed as an Atoms list
                or Trajectory filename.
            restart: bool
                Whether to restart the active learning.
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            comm: MPI communicator.
                The MPI communicator.
        """
        # Setup the ASE calculator
        self.ase_calc = ase_calc
        # Set the initial parameters
        self.reset()
        # Setup the method
        self.setup_method(method)
        # Setup the ML calculator
        self.setup_mlcalc(
            mlcalc,
            save_memory=save_memory,
            verbose=verbose,
            **default_mlcalc_kwargs,
        )
        # Setup the acquisition function
        self.setup_acq(
            acq,
            is_minimization=is_minimization,
            unc_convergence=unc_convergence,
        )
        # Set the arguments
        self.update_arguments(
            is_minimization=is_minimization,
            use_database_check=use_database_check,
            data_perturb=data_perturb,
            data_tol=data_tol,
            save_memory=save_memory,
            parallel_run=parallel_run,
            copy_calc=copy_calc,
            verbose=verbose,
            apply_constraint=apply_constraint,
            force_consistent=force_consistent,
            scale_fmax=scale_fmax,
            use_fmax_convergence=use_fmax_convergence,
            unc_convergence=unc_convergence,
            use_method_unc_conv=use_method_unc_conv,
            use_restart=use_restart,
            check_unc=check_unc,
            check_energy=check_energy,
            check_fmax=check_fmax,
            max_unc_restart=max_unc_restart,
            n_evaluations_each=n_evaluations_each,
            min_data=min_data,
            save_properties_traj=save_properties_traj,
            to_save_mlcalc=to_save_mlcalc,
            save_mlcalc_kwargs=save_mlcalc_kwargs,
            trajectory=trajectory,
            trainingset=trainingset,
            pred_evaluated=pred_evaluated,
            converged_trajectory=converged_trajectory,
            initial_traj=initial_traj,
            tabletxt=tabletxt,
            timetxt=timetxt,
            seed=seed,
            dtype=dtype,
            comm=comm,
            **kwargs,
        )
        # Restart the active learning
        prev_calculations = self.restart_optimization(
            restart,
            prev_calculations,
        )
        # Use previous calculations to train ML calculator
        self.use_prev_calculations(prev_calculations)

    def run(
        self,
        fmax=0.05,
        steps=200,
        ml_steps=1000,
        max_unc=0.3,
        dtrust=None,
        **kwargs,
    ):
        """
        Run the active learning optimization.

        Parameters:
            fmax: float
                Convergence criteria (in eV/Angs).
            steps: int
                Maximum number of evaluations.
            ml_steps: int
                Maximum number of steps for the optimization method
                on the predicted landscape.
            max_unc: float (optional)
                Maximum uncertainty for continuation of the optimization.
                If max_unc is None, then the optimization is performed
                without the maximum uncertainty.
            dtrust: float (optional)
                The trust distance for the optimization method.

        Returns:
            converged: bool
                Whether the active learning is converged.
        """
        # Check if there are any training data
        self.extra_initial_data()
        # Run the active learning
        for step in range(1, steps + 1):
            # Check if the method is converged
            if self.converged():
                self.message_system("Active learning is converged.")
                self.save_trajectory(
                    self.converged_trajectory,
                    self.best_structures,
                    mode="w",
                )
                break
            # Train and optimize ML model
            self.train_mlmodel()
            # Run the method
            candidates, method_converged = self.find_next_candidates(
                fmax=self.scale_fmax * fmax,
                step=step,
                ml_steps=ml_steps,
                max_unc=max_unc,
                dtrust=dtrust,
            )
            # Evaluate candidate
            self.evaluate_candidates(candidates)
            # Print the results for this iteration
            self.print_statement()
            # Check for convergence
            self._converged = self.check_convergence(
                fmax,
                method_converged,
            )
        # State if the active learning did not converge
        if not self.converged():
            self.message_system("Active learning did not converge!")
        # Return and broadcast the best atoms
        self.broadcast_best_structures()
        return self.converged()

    def converged(self):
        "Whether the active learning is converged."
        return self._converged

    def get_number_of_steps(self):
        """
        Get the number of steps that have been run.
        """
        return self.steps

    def reset(self, **kwargs):
        """
        Reset the initial parameters for the active learner.
        """
        # Set initial parameters
        self.steps = 0
        self._converged = False
        self.unc = nan
        self.energy_pred = nan
        self.pred_energies = []
        self.uncertainties = []
        self.ml_train_time = nan
        self.method_time = nan
        self.eval_time = nan
        # Set the header for the summary table
        self.make_hdr_table()
        # Set the writing mode
        self.mode = "w"
        return self

    def setup_method(self, method, **kwargs):
        """
        Setup the optimization method.

        Parameters:
            method: OptimizationMethod instance.
                The quantum mechanincal simulation method instance.

        Returns:
            self: The object itself.
        """
        # Save the method
        self.method = method
        # Set the seed for the method
        if hasattr(self, "seed"):
            self.set_method_seed(self.seed)
        # Get the structures
        self.structures = self.get_structures(allow_calculation=False)
        if isinstance(self.structures, list):
            self.n_structures = len(self.structures)
            self.natoms = len(self.structures[0])
        else:
            self.n_structures = 1
            self.natoms = len(self.structures)
        self.best_structures = self.get_structures(allow_calculation=False)
        self._converged = self.method.converged()
        # Set the evaluated candidate and its calculator
        self.candidate = self.get_candidates()[0].copy()
        self.candidate.calc = self.ase_calc
        # Store the best candidate data
        self.bests_data = {
            "atoms": self.candidate.copy(),
            "energy": None,
            "fmax": None,
            "uncertainty": None,
        }
        return self

    def setup_mlcalc(
        self,
        mlcalc=None,
        verbose=True,
        **default_mlcalc_kwargs,
    ):
        """
        Setup the ML calculator.

        Parameters:
            mlcalc: ML-calculator instance (optional)
                The ML-calculator instance used as surrogate surface.
                A default ML-model is used if mlcalc is None.
            verbose: bool
                Whether to print on screen the full output (True) or
                not (False).
            default_mlcalc_kwargs: dict
                The default keyword arguments for the ML calculator.

        Returns:
            self: The object itself.
        """
        # Check if the ML calculator is given
        if mlcalc is not None:
            self.mlcalc = mlcalc
            # Set the verbose for the ML calculator
            if verbose is not None:
                self.set_verbose(verbose=verbose)
        else:
            self.mlcalc = self.setup_default_mlcalc(
                verbose=verbose,
                **default_mlcalc_kwargs,
            )
        # Check if the seed is given
        if hasattr(self, "seed"):
            # Set the seed for the ML calculator
            self.set_mlcalc_seed(self.seed)
        # Check if the dtype is given
        if hasattr(self, "dtype"):
            # Set the dtype for the ML calculator
            self.mlcalc.set_dtype(self.dtype)
        return self

    def setup_default_mlcalc(
        self,
        atoms=None,
        save_memory=False,
        model="tp",
        fp=None,
        baseline=BornRepulsionCalculator(),
        prior=Prior_max(add=1.0),
        use_derivatives=True,
        optimize_hp=True,
        database_reduction=False,
        use_ensemble=False,
        calc_forces=True,
        round_pred=5,
        bayesian=True,
        kappa=2.0,
        reuse_mlcalc_data=False,
        verbose=True,
        calc_kwargs={},
        **mlmodel_kwargs,
    ):
        """
        Setup the ML calculator.

        Parameters:
            atoms: Atoms instance (optional if fp is not None)
                The Atoms instance from the optimization method.
                It is used to setup the fingerprint if it is None.
            save_memory: bool
                Whether to only train the ML calculator and store
                all instances on one CPU.
                If save_memory==True then parallel optimization of
                the hyperparameters can not be achived.
                If save_memory==False no MPI instance is used.
            model: str or Model class instance
                Either the tp that gives the Students T process or
                gp that gives the Gaussian process.
            fp: Fingerprint class instance (optional)
                The fingerprint instance used for the ML model.
                The default InvDistances instance is used if fp is None.
            baseline: Baseline class instance (optional)
                The baseline instance used for the ML model.
                The default is the BornRepulsionCalculator.
            prior: Prior class instance (optional)
                The prior mean instance used for the ML model.
                The default prior is the Prior_max.
            use_derivatives: bool
                Whether to use derivatives of the targets in the ML model.
            optimize_hp: bool
                Whether to optimize the hyperparameters when the model is
                trained.
            database_reduction: bool
                Whether to reduce the training database size.
                A reduction can avoid memory issues and speed up the training.
            use_ensemble: bool
                Whether to use an ensemble model with clustering.
                The use of ensemble models can avoid memory issues and speed up
                the training.
            calc_forces: bool
                Whether to calculate the forces for all energy predictions.
            round_pred: int (optional)
                The number of decimals to round the predictions to.
                If None, the predictions are not rounded.
            bayesian: bool
                Whether to use the Bayesian optimization calculator.
            kappa: float
                The scaling of the uncertainty relative to the energy.
                The uncertainty is added to the predicted energy.
            reuse_mlcalc_data: bool
                Whether to reuse the data from a previous mlcalc.
            verbose: bool
                Whether to print on screen the full output (True) or
                not (False).
            calc_kwargs: dict
                The keyword arguments for the ML calculator.
            mlmodel_kwargs: dict
                Additional keyword arguments for the function
                to create the MLModel instance.

        Returns:
            self: The instance itself.
        """
        # Create the ML calculator
        from ..regression.gp.calculator.default_model import (
            get_default_mlmodel,
        )
        from ..regression.gp.calculator.mlcalc import MLCalculator
        from ..regression.gp.fingerprint.invdistances import InvDistances

        # Check if the save_memory is given
        if save_memory is None:
            try:
                save_memory = self.save_memory
            except NameError:
                raise NameError("The save_memory is not given.")
        # Setup the fingerprint
        if fp is None:
            # Check if the Atoms instance is given
            if atoms is None:
                try:
                    atoms = self.get_structures(
                        get_all=False,
                        allow_calculation=False,
                    )
                except NameError:
                    raise NameError("The Atoms object is not given or stored.")
            # Can only use distances if there are more than one atom
            if len(atoms) > 1:
                if atoms.pbc.any():
                    periodic_softmax = True
                else:
                    periodic_softmax = False
                fp = InvDistances(
                    reduce_dimensions=True,
                    use_derivatives=True,
                    periodic_softmax=periodic_softmax,
                    wrap=True,
                )
        # Setup the ML model
        mlmodel = get_default_mlmodel(
            model=model,
            prior=prior,
            fp=fp,
            baseline=baseline,
            use_derivatives=use_derivatives,
            parallel=(not save_memory),
            optimize_hp=optimize_hp,
            database_reduction=database_reduction,
            use_ensemble=use_ensemble,
            verbose=verbose,
            **mlmodel_kwargs,
        )
        # Get the data from a previous mlcalc if requested and it exist
        if reuse_mlcalc_data:
            if hasattr(self, "mlcalc"):
                data = self.get_data_atoms()
            else:
                data = []
        # Setup the ML calculator
        if bayesian:
            mlcalc = BOCalculator(
                mlmodel=mlmodel,
                calc_forces=calc_forces,
                round_pred=round_pred,
                kappa=kappa,
                **calc_kwargs,
            )
            if not use_derivatives and kappa > 0.0:
                if world.rank == 0:
                    warnings.warn(
                        "The Bayesian optimization calculator "
                        "with a positive kappa value and no derivatives "
                        "is not recommended!"
                    )
        else:
            mlcalc = MLCalculator(
                mlmodel=mlmodel,
                calc_forces=calc_forces,
                round_pred=round_pred,
                **calc_kwargs,
            )
        # Reuse the data from a previous mlcalc if requested
        if reuse_mlcalc_data:
            if len(data):
                mlcalc.add_training(data)
        return mlcalc

    def setup_acq(
        self,
        acq=None,
        is_minimization=True,
        kappa=2.0,
        unc_convergence=0.05,
        **kwargs,
    ):
        """
        Setup the acquisition function.

        Parameters:
            acq: Acquisition class instance.
                The Acquisition instance used for calculating the acq. function
                and choose a candidate to calculate next.
                The default AcqUME instance is used if acq is None.
            is_minimization: bool
                Whether it is a minimization that is performed.
            kappa: float
                The kappa parameter in the acquisition function.
            unc_convergence: float
                Maximum uncertainty for convergence (in eV).
        """
        # Select an acquisition function
        if acq is None:
            # Setup the acquisition function
            if is_minimization:
                from .acquisition import AcqULCB

                self.acq = AcqULCB(
                    objective="min",
                    kappa=kappa,
                    unc_convergence=unc_convergence,
                )
            else:
                from .acquisition import AcqUUCB

                self.acq = AcqUUCB(
                    objective="max",
                    kappa=kappa,
                    unc_convergence=unc_convergence,
                )
        else:
            self.acq = acq.copy()
            # Check if the objective is the same
            objective = self.get_objective_str()
            if acq.objective != objective:
                raise ValueError(
                    "The objective of the acquisition function "
                    "does not match the active learner."
                )
        # Set the seed for the acquisition function
        if hasattr(self, "seed"):
            self.set_acq_seed(self.seed)
        return self

    def get_structures(
        self,
        get_all=True,
        properties=["forces", "energy", "uncertainty"],
        allow_calculation=True,
        **kwargs,
    ):
        """
        Get the list of ASE Atoms object from the method.

        Parameters:
            get_all: bool
                Whether to get all structures or just the first one.
            properties: list of str
                The names of the requested properties.
                If not given, the properties is not calculated.
            allow_calculation: bool
                Whether the properties are allowed to be calculated.

        Returns:
            Atoms object or list of Atoms objects.
        """
        return self.method.get_structures(
            get_all=get_all,
            properties=properties,
            allow_calculation=allow_calculation,
            **kwargs,
        )

    def get_candidates(self):
        """
        Get the list of candidates from the method.
        The candidates are used for the evaluation.

        Returns:
            List of Atoms objects.
        """
        return self.method.get_candidates()

    def copy_candidates(
        self,
        properties=["fmax", "forces", "energy", "uncertainty"],
        allow_calculation=True,
        **kwargs,
    ):
        """
        Get the candidate structure instances with copied properties.
        It is used for active learning.

        Parameters:
            properties: list of str
                The names of the requested properties.
            allow_calculation: bool
                Whether the properties are allowed to be calculated.

        Returns:
            candidates_copy: list of Atoms instances
                The candidates with copied properties.
        """
        return self.method.copy_candidates(
            properties=properties,
            allow_calculation=allow_calculation,
            **kwargs,
        )

    def use_prev_calculations(self, prev_calculations=None, **kwargs):
        """
        Use previous calculations to restart ML calculator.

        Parameters:
            prev_calculations: Atoms list or ASE Trajectory file.
                The user can feed previously calculated data
                for the same hypersurface.
                The previous calculations must be fed as an Atoms list
                or Trajectory filename.
        """
        if prev_calculations is None:
            return self
        if isinstance(prev_calculations, str):
            prev_calculations = read(prev_calculations, ":")
        if isinstance(prev_calculations, list) and len(prev_calculations) == 0:
            return self
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return self

    def update_method(self, structures, **kwargs):
        """
        Update the method with structures.
        Add the ML calculator to the structures in the optimization method.

        Parameters:
            structures: Atoms instance or list of Atoms instances
                The structures that the optimizable instance is dependent on.

        Returns:
            self: The object itself.
        """
        # Initiate the method with given structure(s)
        self.method.update_optimizable(structures)
        # Set the ML calculator in the method
        self.set_mlcalc()
        return self

    def reset_method(self, **kwargs):
        """
        Reset the stps and convergence of the optimization method.
        Add the ML calculator to the structures in the optimization method.
        """
        # Reset the optimization method
        self.method.reset_optimization()
        # Set the ML calculator in the method
        self.set_mlcalc()
        return self

    def set_mlcalc(self, copy_calc=None, **kwargs):
        """
        Set the ML calculator in the method.
        """
        # Set copy_calc if it is not given
        if copy_calc is None:
            copy_calc = self.copy_calc
        # Set the ML calculator in the method
        self.method.set_calculator(self.mlcalc, copy_calc=copy_calc)
        return self

    def get_data_atoms(self, **kwargs):
        """
        Get the list of atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return self.mlcalc.get_data_atoms()

    def update_arguments(
        self,
        method=None,
        ase_calc=None,
        mlcalc=None,
        acq=None,
        is_minimization=None,
        save_memory=None,
        parallel_run=None,
        copy_calc=None,
        verbose=None,
        apply_constraint=None,
        force_consistent=None,
        scale_fmax=None,
        use_fmax_convergence=None,
        unc_convergence=None,
        use_method_unc_conv=None,
        use_restart=None,
        check_unc=None,
        check_energy=None,
        check_fmax=None,
        max_unc_restart=None,
        n_evaluations_each=None,
        min_data=None,
        use_database_check=None,
        data_perturb=None,
        data_tol=None,
        save_properties_traj=None,
        to_save_mlcalc=None,
        save_mlcalc_kwargs=None,
        trajectory=None,
        trainingset=None,
        pred_evaluated=None,
        converged_trajectory=None,
        initial_traj=None,
        tabletxt=None,
        timetxt=None,
        seed=None,
        dtype=None,
        comm=None,
        **kwargs,
    ):
        """
        Update the instance with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            method: OptimizationMethod instance
                The quantum mechanincal simulation method instance.
            ase_calc: ASE calculator instance
                ASE calculator as implemented in ASE.
            mlcalc: ML-calculator instance
                The ML-calculator instance used as surrogate surface.
                The default BOCalculator instance is used if mlcalc is None.
            acq: Acquisition class instance
                The Acquisition instance used for calculating the
                acq. function and choose a candidate to calculate next.
                The default AcqUME instance is used if acq is None.
            is_minimization: bool
                Whether it is a minimization that is performed.
                Alternative is a maximization.
            save_memory: bool
                Whether to only train the ML calculator and store all objects
                on one CPU.
                If save_memory==True then parallel optimization of
                the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.
            parallel_run: bool
                Whether to run method in parallel on multiple CPUs (True) or
                in sequence on 1 CPU (False).
            copy_calc: bool
                Whether to copy the calculator for each candidate
                in the method.
            verbose: bool
                Whether to print on screen the full output (True) or
                not (False).
            apply_constraint: bool
                Whether to apply the constrains of the ASE Atoms instance
                to the calculated forces.
                By default (apply_constraint=True) forces are 0 for
                constrained atoms and directions.
            force_consistent: bool or None.
                Use force-consistent energy calls (as opposed to the energy
                extrapolated to 0 K).
                By default force_consistent=False.
            scale_fmax: float
                The scaling of the fmax for the ML-NEB runs.
                It makes the path converge tighter on surrogate surface.
            use_fmax_convergence: bool
                Whether to use the maximum force as an convergence criterion.
            unc_convergence: float
                Maximum uncertainty for convergence in
                the active learning (in eV).
            use_method_unc_conv: bool
                Whether to use the unc_convergence as a convergence criterion
                in the optimization method.
            use_restart: bool
                Use the result from last robust iteration.
            check_unc: bool
                Check if the uncertainty is large for the restarted result and
                if it is then use the previous initial.
            check_energy: bool
                Check if the energy is larger for the restarted result than
                the previous.
            check_fmax: bool
                Check if the maximum force is larger for the restarted result
                than the initial interpolation and if so then replace it.
            max_unc_restart: float (optional)
                Maximum uncertainty (in eV) for using the structure(s) as
                the restart in the optimization method.
                If max_unc_restart is None, then the optimization is performed
                without the maximum uncertainty.
            n_evaluations_each: int
                Number of evaluations for each iteration.
            min_data: int
                The minimum number of data points in the training set before
                the active learning can converge.
            use_database_check: bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled.
                Please be aware that the predicted structure will differ from
                the structure in the database if the rattling is applied.
            data_perturb: float
                The perturbation of the data structure if it is in the database
                and use_database_check is True.
                data_perturb is the standard deviation of the normal
                distribution used to rattle the structure.
            data_tol: float
                The tolerance for the data structure if it is in the database
                and use_database_check is True.
            save_properties_traj: bool
                Whether to save the calculated properties to the trajectory.
            to_save_mlcalc: bool
                Whether to save the ML calculator to a file after training.
            save_mlcalc_kwargs: dict
                Arguments for saving the ML calculator, like the filename.
            trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the predicted data.
                Or the TrajectoryWriter instance to store the predicted data.
            trainingset: str or TrajectoryWriter instance
                Trajectory filename to store the evaluated training data.
                Or the TrajectoryWriter instance to store the evaluated
                training data.
            pred_evaluated: str or TrajectoryWriter instance (optional)
                Trajectory filename to store the evaluated training data
                with predicted properties.
                Or the TrajectoryWriter instance to store the evaluated
                training data with predicted properties.
                If pred_evaluated is None, then the predicted data is
                not saved.
            converged_trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the converged structure(s).
                Or the TrajectoryWriter instance to store the converged
                structure(s).
            initial_traj: str or TrajectoryWriter instance
                Trajectory filename to store the initial structure(s).
                Or the TrajectoryWriter instance to store the initial
                structure(s).
            tabletxt: str (optional)
                Name of the .txt file where the summary table is printed.
                It is not saved to the file if tabletxt=None.
            timetxt: str (optional)
                Name of the .txt file where the time table is printed.
                It is not saved to the file if timetxt=None.
            prev_calculations: Atoms list or ASE Trajectory file.
                The user can feed previously calculated data
                for the same hypersurface.
                The previous calculations must be fed as an Atoms list
                or Trajectory filename.
            restart: bool
                Whether to restart the active learning.
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
            dtype: type
                The data type of the arrays.
            comm: MPI communicator.
                The MPI communicator.

        Returns:
            self: The updated object itself.
        """
        # Set parallelization
        if save_memory is not None:
            self.save_memory = save_memory
        if comm is not None or not hasattr(self, "comm"):
            # Setup parallelization
            self.parallel_setup(comm)
        if parallel_run is not None:
            self.parallel_run = parallel_run
            if self.parallel_run and self.save_memory:
                raise ValueError(
                    "The save_memory and parallel_run can not "
                    "be True at the same time!"
                )
        # Set the verbose
        if verbose is not None:
            # Whether to have the full output
            self.set_verbose(verbose=verbose)
        elif not hasattr(self, "verbose"):
            self.set_verbose(verbose=False)
        # Set parameters
        if is_minimization is not None:
            self.is_minimization = is_minimization
        if use_database_check is not None:
            self.use_database_check = use_database_check
        if data_perturb is not None:
            self.data_perturb = abs(float(data_perturb))
        if data_tol is not None:
            self.data_tol = abs(float(data_tol))
        if self.use_database_check:
            if self.data_perturb < self.data_tol:
                self.message_system(
                    "It is not recommended that the data_perturb "
                    "is smaller than the data_tol.",
                    is_warning=True,
                )
        if copy_calc is not None:
            self.copy_calc = copy_calc
        if apply_constraint is not None:
            self.apply_constraint = apply_constraint
        elif not hasattr(self, "apply_constraint"):
            self.apply_constraint = True
        if force_consistent is not None:
            self.force_consistent = force_consistent
        elif not hasattr(self, "force_consistent"):
            self.force_consistent = False
        if scale_fmax is not None:
            self.scale_fmax = abs(float(scale_fmax))
        if use_fmax_convergence is not None:
            self.use_fmax_convergence = use_fmax_convergence
        if unc_convergence is not None:
            self.unc_convergence = abs(float(unc_convergence))
        if use_method_unc_conv is not None:
            self.use_method_unc_conv = use_method_unc_conv
        if use_restart is not None:
            self.use_restart = use_restart
        if check_unc is not None:
            self.check_unc = check_unc
        if check_energy is not None:
            self.check_energy = check_energy
        if check_fmax is not None:
            self.check_fmax = check_fmax
        if max_unc_restart is not None:
            self.max_unc_restart = abs(float(max_unc_restart))
        if n_evaluations_each is not None:
            self.n_evaluations_each = int(abs(n_evaluations_each))
            if self.n_evaluations_each < 1:
                self.n_evaluations_each = 1
        if min_data is not None:
            self.min_data = int(abs(min_data))
        if save_properties_traj is not None:
            self.save_properties_traj = save_properties_traj
        if to_save_mlcalc is not None:
            self.to_save_mlcalc = to_save_mlcalc
        if save_mlcalc_kwargs is not None:
            self.save_mlcalc_kwargs = save_mlcalc_kwargs
        if trajectory is not None or not hasattr(self, "trajectory"):
            self.trajectory = trajectory
        if trainingset is not None or not hasattr(self, "trainingset"):
            self.trainingset = trainingset
        if pred_evaluated is not None or not hasattr(self, "pred_evaluated"):
            self.pred_evaluated = pred_evaluated
        if converged_trajectory is not None or not hasattr(
            self, "converged_trajectory"
        ):
            self.converged_trajectory = converged_trajectory
        if initial_traj is not None or not hasattr(self, "initial_traj"):
            self.initial_traj = initial_traj
        if tabletxt is not None:
            self.tabletxt = str(tabletxt)
        elif not hasattr(self, "tabletxt"):
            self.tabletxt = None
        if timetxt is not None:
            self.timetxt = str(timetxt)
        elif not hasattr(self, "timetxt"):
            self.timetxt = None
        # Set ASE calculator
        if ase_calc is not None:
            self.ase_calc = ase_calc
            if method is None:
                self.setup_method(self.method)
        # Update the optimization method
        if method is not None:
            self.setup_method(method)
        # Set the machine learning calculator
        if mlcalc is not None:
            self.setup_mlcalc(mlcalc)
        # Set the acquisition function
        if acq is not None:
            self.setup_acq(
                acq,
                is_minimization=self.is_minimization,
                unc_convergence=self.unc_convergence,
            )
        # Set the seed
        if seed is not None or not hasattr(self, "seed"):
            self.set_seed(seed)
        # Set the data type
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype)
        # Check if the method and BO is compatible
        self.check_attributes()
        return self

    def find_next_candidates(
        self,
        fmax=0.05,
        step=1,
        ml_steps=200,
        max_unc=None,
        dtrust=None,
        **kwargs,
    ):
        "Run the method on the ML surrogate surface."
        # Convergence of the NEB
        method_converged = False
        # Check if the method is running in parallel
        if not self.parallel_run and self.rank != 0:
            return None, method_converged
        # Check if the previous structure were better
        self.initiate_structure(step=step)
        # Run the method
        method_converged = self.run_method(
            fmax=fmax,
            ml_steps=ml_steps,
            max_unc=max_unc,
            dtrust=dtrust,
        )
        # Get the candidates
        candidates = self.choose_candidates()
        return candidates, method_converged

    def run_method(
        self,
        fmax=0.05,
        ml_steps=750,
        max_unc=None,
        dtrust=None,
        **kwargs,
    ):
        "Run the method on the surrogate surface."
        # Set the uncertainty convergence for the method
        if self.use_method_unc_conv:
            unc_convergence = self.unc_convergence
        else:
            unc_convergence = None
        # Start the method time
        self.method_time = time()
        # Run the method
        self.method.run(
            fmax=fmax,
            steps=ml_steps,
            max_unc=max_unc,
            dtrust=dtrust,
            unc_convergence=unc_convergence,
            **kwargs,
        )
        # Store the method time
        self.method_time = time() - self.method_time
        # Check if the method converged
        method_converged = self.method.converged()
        # Get the atoms from the method run
        self.structures = self.get_structures()
        # Write atoms to trajectory
        self.save_trajectory(self.trajectory, self.structures, mode=self.mode)
        return method_converged

    def initiate_structure(self, step=1, **kwargs):
        "Initiate the method with right structure."
        # Define boolean for using the temporary structure
        use_tmp = True
        # Do not use the temporary structure
        if not self.use_restart or step == 1:
            self.message_system("The initial structure is used.")
            use_tmp = False
        # Reuse the temporary structure if it passes tests
        if use_tmp:
            self.update_method(self.structures)
            # Get uncertainty and fmax
            uncmax_tmp, energy_tmp, fmax_tmp = self.get_predictions()
            # Check uncertainty is low enough
            if self.check_unc:
                if uncmax_tmp > self.max_unc_restart:
                    self.message_system(
                        "The uncertainty is too large to "
                        "use the last structure."
                    )
                    use_tmp = False
        # Check fmax is lower than previous structure
        if use_tmp and (self.check_fmax or self.check_energy):
            self.update_method(self.best_structures)
            _, energy_best, fmax_best = self.get_predictions()
            if self.check_fmax:
                if fmax_tmp > fmax_best:
                    self.message_system(
                        "The fmax is too large to use the last structure."
                    )
                    use_tmp = False
            if use_tmp and self.check_energy:
                if energy_tmp > energy_best:
                    self.message_system(
                        "The energy is too large to use the last structure."
                    )
                    use_tmp = False
        # Check if the temporary structure passed the tests
        if use_tmp:
            self.update_method(self.structures)
            self.message_system("The last structure is used.")
        else:
            self.update_method(self.best_structures)
        # Store the best structures with the ML calculator
        self.copy_best_structures()
        # Save the initial trajectory
        if step == 1 and self.initial_traj is not None:
            self.save_trajectory(self.initial_traj, self.best_structures)
        return

    def get_predictions(self, **kwargs):
        "Get the maximum uncertainty, energy, and fmax prediction."
        uncmax = None
        energy = None
        fmax = None
        if self.check_unc:
            uncmax = self.method.get_uncertainty()
        if self.check_energy:
            energy = self.method.get_potential_energy()
        if self.check_fmax:
            fmax = max_(self.method.get_fmax())
        return uncmax, energy, fmax

    def get_candidate_predictions(self, candidates, **kwargs):
        """
        Get the energies, uncertainties, and fmaxs with the ML calculator
        for the candidates.
        """
        energies = []
        uncertainties = []
        fmaxs = []
        for candidate in candidates:
            energies.append(self.get_true_predicted_energy(candidate))
            uncertainties.append(candidate.calc.results["uncertainty"])
            fmaxs.append(sqrt((candidate.get_forces() ** 2).sum(axis=1).max()))
        return (
            asarray(energies).reshape(-1),
            asarray(uncertainties).reshape(-1),
            asarray(fmaxs).reshape(-1),
        )

    def parallel_setup(self, comm, **kwargs):
        "Setup the parallelization."
        if comm is None:
            self.comm = world
        else:
            self.comm = comm
        self.rank = self.comm.rank
        self.size = self.comm.size
        return self

    def remove_parallel_setup(self):
        "Remove the parallelization by removing the communicator."
        self.comm = None
        self.rank = 0
        self.size = 1
        return self

    def add_training(self, atoms_list, **kwargs):
        "Add atoms_list data to ML model on rank=0."
        self.mlcalc.add_training(atoms_list)
        return self.mlcalc

    def train_mlmodel(self, point_interest=None, **kwargs):
        "Train the ML model"
        # Start the training time
        self.ml_train_time = time()
        # Check if the model should be trained on all CPUs
        if self.save_memory:
            if self.rank != 0:
                return self.mlcalc
        # Update database with the points of interest
        if point_interest is not None:
            self.update_database_arguments(point_interest=point_interest)
        else:
            self.update_database_arguments(point_interest=self.best_structures)
        # Train the ML model
        self.mlcalc.train_model()
        # Store the training time
        self.ml_train_time = time() - self.ml_train_time
        # Save the ML calculator if requested
        if self.to_save_mlcalc:
            self.save_mlcalc(**self.save_mlcalc_kwargs)
        return self.mlcalc

    def save_data(self, **kwargs):
        "Save the training data to a file."
        if self.steps > 1:
            self.mlcalc.save_data(
                trajectory=self.trainingset,
                mode="a",
                write_last=True,
                **kwargs,
            )
        else:
            self.mlcalc.save_data(trajectory=self.trainingset, **kwargs)
        return self

    def save_trajectory(self, trajectory, structures, mode="w", **kwargs):
        "Save the trajectory of the data."
        if trajectory is None:
            return self
        if isinstance(trajectory, str):
            with TrajectoryWriter(trajectory, mode=mode) as traj:
                self.save_traj(traj, structures, **kwargs)
        elif isinstance(trajectory, TrajectoryWriter):
            self.save_traj(trajectory, structures, **kwargs)
        else:
            self.message_system(
                "The trajectory type is not supported. "
                "The trajectory is not saved!"
            )
        return self

    def save_traj(self, traj, structures, **kwargs):
        "Save the trajectory of the data with the TrajectoryWriter."
        if not isinstance(structures, list):
            structures = [structures]
        for struc in structures:
            if struc is not None:
                if self.save_properties_traj:
                    if hasattr(struc.calc, "results"):
                        struc.info["results"] = struc.calc.results
                    else:
                        struc.info["results"] = {}
                traj.write(struc)
        return traj

    def evaluate_candidates(self, candidates, **kwargs):
        "Evaluate the candidates."
        # Check if the candidates are a list
        if not isinstance(candidates, (list, ndarray)):
            candidates = [candidates]
        # Evaluate the candidates
        for candidate in candidates:
            # Ensure that the candidate is not already in the database
            if self.use_database_check:
                candidate = self.ensure_candidate_not_in_database(
                    candidate,
                    show_message=True,
                )
            # Broadcast the predictions
            self.broadcast_predictions()
            # Evaluate the candidate
            self.evaluate(candidate, is_predicted=True)
            # Set the mode to append
            self.mode = "a"
        return self

    def evaluate(self, candidate, is_predicted=False, **kwargs):
        "Evaluate the ASE atoms with the ASE calculator."
        # Ensure that the candidate is not already in the database
        if self.use_database_check and not is_predicted:
            candidate, _ = self.ensure_not_in_database(candidate)
        # Update the evaluated candidate
        self.update_candidate(candidate)
        # Start the evaluation time
        self.eval_time = time()
        # Calculate the energies and forces
        self.message_system("Performing evaluation.", end="\r")
        forces = self.candidate.get_forces(
            apply_constraint=self.apply_constraint
        )
        self.energy_true = self.candidate.get_potential_energy(
            force_consistent=self.force_consistent
        )
        self.message_system("Single-point calculation finished.")
        # Store the evaluation time
        self.eval_time = time() - self.eval_time
        # Save deviation, fmax, and update steps
        self.e_dev = abs(self.energy_true - self.energy_pred)
        self.true_fmax = nanmax(norm(forces, axis=1))
        self.steps += 1
        # Store the data
        if is_predicted:
            # Store the candidate with predicted properties
            self.save_trajectory(
                self.pred_evaluated,
                candidate,
                mode=self.mode,
            )
        self.add_training([self.candidate])
        self.save_data()
        # Make a reference energy
        if self.steps == 1:
            atoms_ref = self.get_data_atoms()[0]
            self.e_ref = atoms_ref.get_potential_energy()
        # Store the best evaluated candidate
        self.store_best_data(self.candidate)
        # Make the summary table
        self.make_summary_table()
        return

    def update_candidate(self, candidate, dtol=1e-8, **kwargs):
        "Update the evaluated candidate with given candidate."
        # Broadcast the system to all cpus
        if self.rank == 0:
            candidate = candidate.copy()
        candidate = broadcast(candidate, root=0, comm=self.comm)
        # Update the evaluated candidate with given candidate
        # Set positions
        self.candidate.set_positions(candidate.get_positions())
        # Set cell
        cell_old = self.candidate.get_cell()
        cell_new = candidate.get_cell()
        if norm(cell_old - cell_new) > dtol:
            self.candidate.set_cell(cell_new)
        # Set pbc
        pbc_old = self.candidate.get_pbc()
        pbc_new = candidate.get_pbc()
        if (pbc_old == pbc_new).all():
            self.candidate.set_pbc(pbc_new)
        # Set initial charges
        ini_charge_old = self.candidate.get_initial_charges()
        ini_charge_new = candidate.get_initial_charges()
        if norm(ini_charge_old - ini_charge_new) > dtol:
            self.candidate.set_initial_charges(ini_charge_new)
        # Set initial magmoms
        ini_magmom_old = self.candidate.get_initial_magnetic_moments()
        ini_magmom_new = candidate.get_initial_magnetic_moments()
        if norm(ini_magmom_old - ini_magmom_new) > dtol:
            self.candidate.set_initial_magnetic_moments(ini_magmom_new)
        # Set momenta
        momenta_old = self.candidate.get_momenta()
        momenta_new = candidate.get_momenta()
        if norm(momenta_old - momenta_new) > dtol:
            self.candidate.set_momenta(momenta_new)
        # Set velocities
        velocities_old = self.candidate.get_velocities()
        velocities_new = candidate.get_velocities()
        if norm(velocities_old - velocities_new) > dtol:
            self.candidate.set_velocities(velocities_new)
        return candidate

    def broadcast_predictions(self, **kwargs):
        "Broadcast the predictions."
        # Get energy and uncertainty and remove it from the list
        if self.rank == 0:
            self.energy_pred = self.pred_energies[0]
            self.pred_energies = self.pred_energies[1:]
            self.unc = self.uncertainties[0]
            self.uncertainties = self.uncertainties[1:]
        # Broadcast the predictions
        self.energy_pred = broadcast(self.energy_pred, root=0, comm=self.comm)
        self.unc = broadcast(self.unc, root=0, comm=self.comm)
        self.pred_energies = broadcast(
            self.pred_energies,
            root=0,
            comm=self.comm,
        )
        self.uncertainties = broadcast(
            self.uncertainties,
            root=0,
            comm=self.comm,
        )
        return self

    def extra_initial_data(self, **kwargs):
        """
        Get an initial structure for the active learning
        if the ML calculator does not have any training points.
        """
        # Get the number of training data
        n_data = self.get_training_set_size()
        # Check if the training set is empty
        if n_data >= 2:
            return self
        # Get the initial structure
        atoms = self.get_structures(get_all=False, allow_calculation=False)
        # Rattle if the initial structure is calculated
        if n_data == 1:
            atoms = self.rattle_atoms(atoms, data_perturb=0.02)
        # Evaluate the structure
        self.evaluate(atoms)
        # Print summary table
        self.print_statement()
        # Check if another initial data is needed
        if n_data == 0:
            self.extra_initial_data(**kwargs)
        return self

    def update_database_arguments(self, point_interest=None, **kwargs):
        "Update the arguments in the database."
        self.mlcalc.update_database_arguments(
            point_interest=point_interest,
            **kwargs,
        )
        return self

    def ensure_not_in_database(
        self,
        atoms,
        show_message=True,
        **kwargs,
    ):
        "Ensure the ASE Atoms instance is not in database by perturb it."
        # Return atoms if it does not exist
        if atoms is None:
            return atoms
        # Boolean for checking if the atoms instance was in database
        was_in_database = False
        # Check if atoms instance is in the database
        while self.is_in_database(atoms, dtol=self.data_tol, **kwargs):
            # Atoms instance was in database
            was_in_database = True
            # Rattle the atoms
            atoms = self.rattle_atoms(atoms, data_perturb=self.data_perturb)
            # Print message if requested
            if show_message:
                self.message_system(
                    "The system is rattled, since it is already in "
                    "the database."
                )
        return atoms, was_in_database

    def rattle_atoms(self, atoms, data_perturb, **kwargs):
        "Rattle the ASE Atoms instance positions."
        # Get positions
        pos = atoms.get_positions()
        # Rattle the positions
        pos_new = pos + self.rng.normal(
            loc=0.0,
            scale=data_perturb,
            size=pos.shape,
        )
        # Set the new positions
        atoms.set_positions(pos_new)
        return atoms

    def ensure_candidate_not_in_database(
        self,
        candidate,
        show_message=True,
        **kwargs,
    ):
        "Ensure the candidate is not in database by perturb it."
        # Check if the method is running in parallel
        if not self.parallel_run and self.rank != 0:
            return None
        # Ensure that the candidate is not already in the database
        candidate, was_in_database = self.ensure_not_in_database(
            candidate,
            show_message=show_message,
        )
        # Calculate the properties if it was in the database
        if was_in_database:
            candidate.calc = self.mlcalc
            candidate = self.method.copy_atoms(
                candidate,
                properties=["fmax", "uncertainty", "energy"],
                allow_calculation=True,
            )
            self.pred_energies[0] = self.get_true_predicted_energy(candidate)
            self.uncertainties[0] = candidate.calc.results["uncertainty"]
        return candidate

    def store_best_data(self, atoms, **kwargs):
        "Store the best candidate."
        update = True
        # Check if the energy is better than the previous best
        if self.is_minimization:
            best_energy = self.bests_data["energy"]
            if best_energy is not None and self.energy_true > best_energy:
                update = False
        # Update the best data
        if update:
            self.bests_data["atoms"] = atoms.copy()
            self.bests_data["energy"] = self.energy_true
            self.bests_data["fmax"] = self.true_fmax
            self.bests_data["uncertainty"] = self.unc
        return self

    def get_training_set_size(self):
        "Get the size of the training set"
        return self.mlcalc.get_training_set_size()

    def choose_candidates(self, **kwargs):
        "Use acquisition functions to chose the next training points"
        # Get the candidates
        candidates = self.copy_candidates()
        # Get the energies and uncertainties
        energies, uncertainties, fmaxs = self.get_candidate_predictions(
            candidates
        )
        # Store the uncertainty predictions
        self.umax = max_(uncertainties)
        self.umean = mean_(uncertainties)
        # Calculate the acquisition function for each candidate
        acq_values = self.acq.calculate(
            energy=energies,
            uncertainty=uncertainties,
            fmax=fmaxs,
        )
        # Chose the candidates given by the Acq. class
        i_cand = self.acq.choose(acq_values)
        i_cand = i_cand[: self.n_evaluations_each]
        # Reverse the order of the candidates so the best is last
        if self.n_evaluations_each > 1:
            i_cand = i_cand[::-1]
        # The next training points
        candidates = [candidates[i] for i in i_cand]
        self.pred_energies = energies[i_cand]
        self.uncertainties = uncertainties[i_cand]
        return candidates

    def check_convergence(self, fmax, method_converged, **kwargs):
        "Check if the convergence criteria are fulfilled"
        converged = True
        if self.rank == 0:
            # Check if the method converged
            if not method_converged:
                converged = False
            # Check if the minimum number of trained data points is reached
            if self.get_training_set_size() - 1 < self.min_data:
                converged = False
            # Check the force criterion is met if it is requested
            if self.use_fmax_convergence and self.true_fmax > fmax:
                converged = False
            # Check the uncertainty criterion is met
            if self.umax > self.unc_convergence:
                converged = False
            # Check the true energy deviation
            # match the uncertainty prediction
            uci = 2.0 * self.unc_convergence
            if self.e_dev > uci:
                converged = False
            # Check if the energy is the minimum
            if self.is_minimization:
                e_dif = abs(self.energy_true - self.bests_data["energy"])
                if e_dif > uci:
                    converged = False
        # Broadcast convergence statement if MPI is used
        converged = broadcast(converged, root=0, comm=self.comm)
        # Check the convergence
        if converged:
            self.copy_best_structures()
        return converged

    def copy_best_structures(
        self,
        get_all=True,
        properties=["forces", "energy", "uncertainty"],
        allow_calculation=True,
        **kwargs,
    ):
        """
        Copy the best structures.

        Parameters:
            properties: list of str
                The names of the requested properties.
                If not given, the properties is not calculated.
            allow_calculation: bool
                Whether the properties are allowed to be calculated.

        Returns:
            list of ASE Atoms objects: The best structures.
        """
        # Check if the method is running in parallel
        if not self.parallel_run and self.rank != 0:
            return self.best_structures
        # Get the best structures with calculated properties
        self.best_structures = self.get_structures(
            get_all=get_all,
            properties=properties,
            allow_calculation=allow_calculation,
            **kwargs,
        )
        return self.best_structures

    def get_best_structures(self):
        "Get the best structures."
        return self.best_structures

    def broadcast_best_structures(self):
        "Broadcast the best structures."
        self.best_structures = broadcast(
            self.best_structures,
            root=0,
            comm=self.comm,
        )
        return self.best_structures

    def copy_atoms(self, atoms):
        "Copy the ASE Atoms instance with calculator."
        return copy_atoms(atoms)

    def compare_atoms(
        self,
        atoms0,
        atoms1,
        tol=1e-8,
        properties_to_check=["atoms", "positions", "cell", "pbc"],
        **kwargs,
    ):
        """
        Compare two ASE Atoms instances.
        """
        is_same = compare_atoms(
            atoms0,
            atoms1,
            tol=tol,
            properties_to_check=properties_to_check,
            **kwargs,
        )
        return is_same

    def get_objective_str(self, **kwargs):
        "Get what the objective is for the active learning."
        if not self.is_minimization:
            return "max"
        return "min"

    def set_verbose(self, verbose, **kwargs):
        "Set verbose of MLModel."
        self.verbose = verbose
        self.mlcalc.mlmodel.update_arguments(verbose=verbose)
        return self

    def is_in_database(self, atoms, **kwargs):
        "Check if the ASE Atoms is in the database."
        return self.mlcalc.is_in_database(atoms, **kwargs)

    def get_true_predicted_energy(self, atoms, **kwargs):
        """
        Get the true predicted energy of the atoms.
        Since the BOCalculator will return the predicted energy and
        the uncertainty times the kappa value, this should be avoided.
        """
        energy = atoms.get_potential_energy()
        if hasattr(atoms.calc, "results"):
            if "predicted energy" in atoms.calc.results:
                energy = atoms.calc.results["predicted energy"]
        return energy

    def save_mlcalc(self, filename="mlcalc.pkl", **kwargs):
        """
        Save the ML calculator object to a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            self: The object itself.
        """
        self.mlcalc.save_mlcalc(filename, **kwargs)
        return self

    def get_mlcalc(self, copy_mlcalc=True, **kwargs):
        """
        Get the ML calculator instance.

        Parameters:
            copy_mlcalc: bool
                Whether to copy the instance.

        Returns:
            MLCalculator: The ML calculator instance.
        """
        if copy_mlcalc:
            return self.mlcalc.copy()
        return self.mlcalc

    def check_attributes(self, **kwargs):
        """
        Check that the active learning and the method
        agree upon the attributes.
        """
        if self.parallel_run != self.method.parallel_run:
            raise ValueError(
                "Active learner and Optimization method does "
                "not agree whether to run in parallel!"
            )
        return self

    def set_seed(self, seed=None, **kwargs):
        """
        Set the random seed.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        if seed is not None:
            self.seed = seed
            if isinstance(seed, int):
                self.rng = default_rng(self.seed)
            elif isinstance(seed, Generator) or isinstance(seed, RandomState):
                self.rng = seed
        else:
            self.seed = None
            self.rng = default_rng()
        # Set the random seed for the optimization method
        self.set_method_seed(self.seed)
        # Set the random seed for the acquisition function
        self.set_acq_seed(self.seed)
        # Set the random seed for the ML calculator
        self.set_mlcalc_seed(self.seed)
        return self

    def set_method_seed(self, seed=None, **kwargs):
        """
        Set the random seed for the optimization method.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        self.method.set_seed(seed)
        return self

    def set_acq_seed(self, seed=None, **kwargs):
        """
        Set the random seed for the acquisition function.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        self.acq.set_seed(seed)
        return self

    def set_mlcalc_seed(self, seed=None, **kwargs):
        """
        Set the random seed for the ML calculator.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        self.mlcalc.set_seed(seed)
        return self

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the data type
        self.dtype = dtype
        # Set the data type of the mlcalc
        self.mlcalc.set_dtype(dtype)
        return self

    def set_kappa(self, kappa, **kwargs):
        """
        Set the kappa value for the acquisition function.
        Kappa is used to scale the uncertainty in the acquisition function.
        Furthermore, set the kappa value for the ML calculator
        if it is a BOCalculator.

        Parameters:
            kappa: float
                The kappa value to set for the acquisition function and
                the ML calculator.

        Returns:
            self: The instance itself.
        """
        # Set the kappa value for the acquisition function
        self.acq.set_kappa()
        # Set the kappa value for the ML calculator if it is a BOCalculator
        if isinstance(self.mlcalc, BOCalculator):
            self.mlcalc.set_kappa(kappa)
        return self

    def message_system(self, message, obj=None, end="\n", is_warning=False):
        "Print output once."
        if self.verbose is True and self.rank == 0:
            if is_warning:
                warnings.warn(message)
            else:
                if obj is None:
                    print(message, end=end)
                else:
                    print(message, obj, end=end)
        return

    def make_hdr_table(self, **kwargs):
        "Make the header of the summary tables for the optimization process."
        # Make the header to the summary table
        hdr_list = [
            " {:<6} ".format("Step"),
            " {:<11s} ".format("Date"),
            " {:<16s} ".format("True energy/[eV]"),
            " {:<16s} ".format("Uncertainty/[eV]"),
            " {:<15s} ".format("True error/[eV]"),
            " {:<16s} ".format("True fmax/[eV/Å]"),
        ]
        # Write the header
        hdr = "|" + "|".join(hdr_list) + "|"
        self.print_list = [hdr]
        # Make the header to the time summary table
        hdr_list = [
            " {:<6} ".format("Step"),
            " {:<11s} ".format("Date"),
            " {:<16s} ".format("ML training/[s]"),
            " {:<16s} ".format("ML run/[s]"),
            " {:<16s} ".format("Evaluation/[s]"),
        ]
        # Write the header to the time summary table
        hdr_time = "|" + "|".join(hdr_list) + "|"
        self.print_list_time = [hdr_time]
        return hdr

    def make_summary_table(self, **kwargs):
        "Make the summary of the optimization process as table."
        if self.rank != 0:
            return None, None
        now = datetime.datetime.now().strftime("%d %H:%M:%S")
        # Make the row for the summary table
        msg = [
            " {:<6d} ".format(self.steps),
            " {:<11s} ".format(now),
            " {:16.4f} ".format(self.energy_true - self.e_ref),
            " {:16.4f} ".format(self.unc),
            " {:15.4f} ".format(self.e_dev),
            " {:16.4f} ".format(self.true_fmax),
        ]
        msg = "|" + "|".join(msg) + "|"
        self.print_list.append(msg)
        msg = "\n".join(self.print_list)
        # Make the row for the time summary table
        msg_time = [
            " {:<6d} ".format(self.steps),
            " {:<11s} ".format(now),
            " {:16.4f} ".format(self.ml_train_time),
            " {:16.4f} ".format(self.method_time),
            " {:16.4f} ".format(self.eval_time),
        ]
        msg_time = "|" + "|".join(msg_time) + "|"
        self.print_list_time.append(msg_time)
        msg_time = "\n".join(self.print_list_time)
        return msg, msg_time

    def save_summary_table(self, msg=None, **kwargs):
        "Save the summary table in the .txt file."
        if self.tabletxt is not None:
            with open(self.tabletxt, "w") as thefile:
                if msg is None:
                    msg = "\n".join(self.print_list)
                thefile.writelines(msg)
        if self.timetxt is not None:
            with open(self.timetxt, "w") as thefile:
                msg = "\n".join(self.print_list_time)
                thefile.writelines(msg)
        return

    def print_statement(self, **kwargs):
        "Print the active learning process as a table."
        msg = ""
        if self.rank == 0:
            msg = "\n".join(self.print_list)
            self.save_summary_table(msg)
            self.message_system(msg)
        return msg

    def restart_optimization(
        self,
        restart=False,
        prev_calculations=None,
        **kwargs,
    ):
        "Restart the active learning."
        # Check if the optimization should be restarted
        if not restart:
            return prev_calculations
        # Load the previous calculations from trajectory
        # Test if the restart is possible
        structure = read(self.trajectory, "0")
        if len(structure) != self.natoms:
            raise ValueError(
                "The number of atoms in the trajectory does not match "
                "the number of atoms in given."
            )
        # Load the predicted structures
        if self.n_structures == 1:
            index = "-1"
        else:
            index = f"-{self.n_structures}:"
        self.structures = read(
            self.trajectory,
            index,
        )
        # Load the previous training data
        prev_calculations = read(self.trainingset, ":")
        # Update the method with the structures
        self.update_method(self.structures)
        # Set the writing mode
        self.mode = "a"
        # Load the summary table
        if self.tabletxt is not None:
            with open(self.tabletxt, "r") as thefile:
                self.print_list = [line.replace("\n", "") for line in thefile]
            # Update the total steps
            self.steps = len(self.print_list) - 1
            # Make a reference energy
            atoms_ref = self.copy_atoms(prev_calculations[0])
            self.e_ref = atoms_ref.get_potential_energy()
        # Load the time summary table
        if self.timetxt is not None:
            with open(self.timetxt, "r") as thefile:
                self.print_list_time = [
                    line.replace("\n", "") for line in thefile
                ]
            # Update the total steps
            if self.tabletxt is None:
                self.steps = len(self.print_list_time) - 1
        return prev_calculations

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            method=self.method,
            ase_calc=self.ase_calc,
            mlcalc=self.mlcalc,
            acq=self.acq,
            is_minimization=self.is_minimization,
            save_memory=self.save_memory,
            parallel_run=self.parallel_run,
            copy_calc=self.copy_calc,
            verbose=self.verbose,
            apply_constraint=self.apply_constraint,
            force_consistent=self.force_consistent,
            scale_fmax=self.scale_fmax,
            use_fmax_convergence=self.use_fmax_convergence,
            unc_convergence=self.unc_convergence,
            use_method_unc_conv=self.use_method_unc_conv,
            use_restart=self.use_restart,
            check_unc=self.check_unc,
            check_energy=self.check_energy,
            check_fmax=self.check_fmax,
            max_unc_restart=self.max_unc_restart,
            n_evaluations_each=self.n_evaluations_each,
            min_data=self.min_data,
            use_database_check=self.use_database_check,
            data_perturb=self.data_perturb,
            data_tol=self.data_tol,
            save_properties_traj=self.save_properties_traj,
            to_save_mlcalc=self.to_save_mlcalc,
            save_mlcalc_kwargs=self.save_mlcalc_kwargs,
            trajectory=self.trajectory,
            trainingset=self.trainingset,
            pred_evaluated=self.pred_evaluated,
            converged_trajectory=self.converged_trajectory,
            initial_traj=self.initial_traj,
            tabletxt=self.tabletxt,
            timetxt=self.timetxt,
            seed=self.seed,
            dtype=self.dtype,
            comm=self.comm,
        )
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
