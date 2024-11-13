import numpy as np
from ase.io import read
from ase.parallel import world, broadcast
from ase.io.trajectory import TrajectoryWriter
import datetime
from ..regression.gp.calculator.copy_atoms import copy_atoms


class ActiveLearning:
    def __init__(
        self,
        method,
        ase_calc,
        mlcalc=None,
        acq=None,
        is_minimization=True,
        use_database_check=True,
        save_memory=False,
        parallel_run=False,
        copy_calc=False,
        verbose=True,
        apply_constraint=True,
        force_consistent=False,
        scale_fmax=0.8,
        use_fmax_convergence=True,
        unc_convergence=0.05,
        use_method_unc_conv=True,
        use_restart=True,
        check_unc=True,
        check_energy=True,
        check_fmax=True,
        n_evaluations_each=1,
        min_data=3,
        save_properties_traj=True,
        trajectory="predicted.traj",
        trainingset="evaluated.traj",
        converged_trajectory="converged.traj",
        initial_traj="initial_struc.traj",
        tabletxt="ml_summary.txt",
        prev_calculations=None,
        restart=False,
        comm=world,
        **kwargs,
    ):
        """
        An active learner that is used for accelerating quantum mechanincal
        simulation methods with an active learning approach.

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
            use_database_check: bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled.
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
            n_evaluations_each: int
                Number of evaluations for each iteration.
            min_data: int
                The minimum number of data points in the training set before
                the active learning can converge.
            save_properties_traj: bool
                Whether to save the calculated properties to the trajectory.
            trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the predicted data.
                Or the TrajectoryWriter instance to store the predicted data.
            trainingset: str or TrajectoryWriter instance
                Trajectory filename to store the evaluated training data.
                Or the TrajectoryWriter instance to store the evaluated
                training data.
            converged_trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the converged structure(s).
                Or the TrajectoryWriter instance to store the converged
                structure(s).
            initial_traj: str or TrajectoryWriter instance
                Trajectory filename to store the initial structure(s).
                Or the TrajectoryWriter instance to store the initial
                structure(s).
            tabletxt: str
                Name of the .txt file where the summary table is printed.
                It is not saved to the file if tabletxt=None.
            prev_calculations: Atoms list or ASE Trajectory file.
                The user can feed previously calculated data
                for the same hypersurface.
                The previous calculations must be fed as an Atoms list
                or Trajectory filename.
            restart: bool
                Whether to restart the active learning.
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
            n_evaluations_each=n_evaluations_each,
            min_data=min_data,
            save_properties_traj=save_properties_traj,
            trajectory=trajectory,
            trainingset=trainingset,
            converged_trajectory=converged_trajectory,
            initial_traj=initial_traj,
            tabletxt=tabletxt,
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
        max_unc=None,
        dtrust=None,
        seed=None,
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
            dtrust: float (optional)
                The trust distance for the optimization method.
            seed: int (optional)
                The random seed.

        Returns:
            converged: bool
                Whether the active learning is converged.
        """
        # Set the random seed
        if seed is not None:
            np.random.seed(seed)
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
        self.unc = np.nan
        self.energy_pred = np.nan
        self.pred_energies = []
        self.uncertainties = []
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
        self.structures = self.get_structures()
        if isinstance(self.structures, list):
            self.n_structures = len(self.structures)
            self.natoms = len(self.structures[0])
        else:
            self.n_structures = 1
            self.natoms = len(self.structures)
        self.best_structures = self.get_structures()
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
        save_memory=False,
        fp=None,
        atoms=None,
        prior=None,
        baseline=None,
        use_derivatives=True,
        database_reduction=False,
        calc_forces=True,
        bayesian=True,
        kappa=2.0,
        reuse_mlcalc_data=False,
        verbose=True,
        **kwargs,
    ):
        """
        Setup the ML calculator.

        Parameters:
            mlcalc: ML-calculator instance (optional)
                The ML-calculator instance used as surrogate surface.
                A default ML-model is used if mlcalc is None.
            save_memory: bool
                Whether to only train the ML calculator and store
                all objects on one CPU.
                If save_memory==True then parallel optimization of
                the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.
            fp: Fingerprint class instance (optional)
                The fingerprint instance used for the ML model.
                The default InvDistances instance is used if fp is None.
            atoms: Atoms object (optional if fp is not None)
                The Atoms object from the optimization method.
                It is used to setup the fingerprint if it is None.
            prior: Prior class instance (optional)
                The prior mean instance used for the ML model.
                The default Prior_max instance is used if prior is None.
            baseline: Baseline class instance (optional)
                The baseline instance used for the ML model.
                The default is None.
            use_derivatives : bool
                Whether to use derivatives of the targets in the ML model.
            database_reduction: bool
                Whether to reduce the database.
            calc_forces: bool
                Whether to calculate the forces for all energy predictions.
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

        Returns:
            self: The object itself.
        """
        # Check if the ML calculator is given
        if mlcalc is not None:
            self.mlcalc = mlcalc
            return self
        # Create the ML calculator
        from ..regression.gp.calculator.mlmodel import get_default_mlmodel
        from ..regression.gp.calculator.bocalc import BOCalculator
        from ..regression.gp.calculator.mlcalc import MLCalculator
        from ..regression.gp.means.max import Prior_max
        from ..regression.gp.fingerprint.invdistances import InvDistances

        # Check if the save_memory is given
        if save_memory is None:
            try:
                save_memory = self.save_memory
            except Exception:
                raise Exception("The save_memory is not given.")
        # Setup the fingerprint
        if fp is None:
            # Check if the Atoms object is given
            if atoms is None:
                try:
                    atoms = self.get_structures(get_all=False)
                except Exception:
                    raise Exception("The Atoms object is not given or stored.")
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
                    wrap=False,
                )
        # Setup the prior mean
        if prior is None:
            prior = Prior_max(add=1.0)
        # Setup the ML model
        mlmodel = get_default_mlmodel(
            model="tp",
            prior=prior,
            fp=fp,
            baseline=baseline,
            use_derivatives=use_derivatives,
            parallel=(not save_memory),
            database_reduction=database_reduction,
            verbose=verbose,
        )
        # Get the data from a previous mlcalc if requested and it exist
        if reuse_mlcalc_data:
            if hasattr(self, "mlcalc"):
                data = self.get_data_atoms()
            else:
                data = []
        # Setup the ML calculator
        if bayesian:
            self.mlcalc = BOCalculator(
                mlmodel=mlmodel,
                calc_forces=calc_forces,
                kappa=kappa,
            )
            if not use_derivatives and kappa > 0.0:
                if world.rank == 0:
                    print(
                        "Warning: The Bayesian optimization calculator "
                        "with a positive kappa value and no derivatives "
                        "is not recommended!"
                    )
        else:
            self.mlcalc = MLCalculator(
                mlmodel=mlmodel,
                calc_forces=calc_forces,
            )
        # Reuse the data from a previous mlcalc if requested
        if reuse_mlcalc_data:
            if len(data):
                self.add_training(data)
        return self

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
            acq : Acquisition class instance.
                The Acquisition instance used for calculating the acq. function
                and choose a candidate to calculate next.
                The default AcqUME instance is used if acq is None.
            is_minimization : bool
                Whether it is a minimization that is performed.
            kappa : float
                The kappa parameter in the acquisition function.
            unc_convergence : float
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
                raise Exception(
                    "The objective of the acquisition function "
                    "does not match the active learner."
                )
        return self

    def get_structures(
        self,
        get_all=True,
        **kwargs,
    ):
        """
        Get the list of ASE Atoms object from the method.

        Parameters:
            get_all : bool
                Whether to get all structures or just the first one.

        Returns:
            Atoms object or list of Atoms objects.
        """
        return self.method.get_structures(get_all=get_all, **kwargs)

    def get_candidates(self):
        """
        Get the list of candidates from the method.
        The candidates are used for the evaluation.

        Returns:
            List of Atoms objects.
        """
        return self.method.get_candidates()

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
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return self

    def update_method(self, structures, **kwargs):
        """
        Update the method with structures.

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
        use_database_check=None,
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
        n_evaluations_each=None,
        min_data=None,
        save_properties_traj=None,
        trajectory=None,
        trainingset=None,
        converged_trajectory=None,
        initial_traj=None,
        tabletxt=None,
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
            use_database_check: bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled.
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
            n_evaluations_each: int
                Number of evaluations for each iteration.
            min_data: int
                The minimum number of data points in the training set before
                the active learning can converge.
            save_properties_traj: bool
                Whether to save the calculated properties to the trajectory.
            trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the predicted data.
                Or the TrajectoryWriter instance to store the predicted data.
            trainingset: str or TrajectoryWriter instance
                Trajectory filename to store the evaluated training data.
                Or the TrajectoryWriter instance to store the evaluated
                training data.
            converged_trajectory: str or TrajectoryWriter instance
                Trajectory filename to store the converged structure(s).
                Or the TrajectoryWriter instance to store the converged
                structure(s).
            initial_traj: str or TrajectoryWriter instance
                Trajectory filename to store the initial structure(s).
                Or the TrajectoryWriter instance to store the initial
                structure(s).
            tabletxt: str
                Name of the .txt file where the summary table is printed.
                It is not saved to the file if tabletxt=None.
            prev_calculations: Atoms list or ASE Trajectory file.
                The user can feed previously calculated data
                for the same hypersurface.
                The previous calculations must be fed as an Atoms list
                or Trajectory filename.
            restart: bool
                Whether to restart the active learning.
            comm: MPI communicator.
                The MPI communicator.

        Returns:
            self: The updated object itself.
        """
        # Fixed parameters
        if is_minimization is not None:
            self.is_minimization = is_minimization
        if use_database_check is not None:
            self.use_database_check = use_database_check
        if save_memory is not None:
            self.save_memory = save_memory
        if comm is not None:
            # Setup parallelization
            self.parallel_setup(comm)
        if parallel_run is not None:
            self.parallel_run = parallel_run
        if copy_calc is not None:
            self.copy_calc = copy_calc
        if verbose is not None:
            # Whether to have the full output
            self.verbose = verbose
            self.set_verbose(verbose=verbose)
        if apply_constraint is not None:
            self.apply_constraint = apply_constraint
        if force_consistent is not None:
            self.force_consistent = force_consistent
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
        if n_evaluations_each is not None:
            self.n_evaluations_each = int(abs(n_evaluations_each))
            if self.n_evaluations_each < 1:
                self.n_evaluations_each = 1
        if min_data is not None:
            self.min_data = int(abs(min_data))
        if save_properties_traj is not None:
            self.save_properties_traj = save_properties_traj
        if trajectory is not None:
            self.trajectory = trajectory
        elif not hasattr(self, "trajectory"):
            self.trajectory = None
        if trainingset is not None:
            self.trainingset = trainingset
        elif not hasattr(self, "trainingset"):
            self.trainingset = None
        if converged_trajectory is not None:
            self.converged_trajectory = converged_trajectory
        elif not hasattr(self, "converged_trajectory"):
            self.converged_trajectory = None
        if initial_traj is not None:
            self.initial_traj = initial_traj
        elif not hasattr(self, "initial_traj"):
            self.initial_traj = None
        if tabletxt is not None:
            self.tabletxt = str(tabletxt)
        elif not hasattr(self, "tabletxt"):
            self.tabletxt = None
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
        # If memeory is saved the method is only performed on one CPU
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
        # Run the method
        self.method.run(
            fmax=fmax,
            steps=ml_steps,
            max_unc=max_unc,
            dtrust=dtrust,
            unc_convergence=unc_convergence,
            **kwargs,
        )
        # Check if the method converged
        method_converged = self.method.converged()
        # Get the atoms from the method run
        self.structures = self.get_structures()
        # Write atoms to trajectory
        self.save_trajectory(self.trajectory, self.structures, mode=self.mode)
        # Set the mode to append
        self.mode = "a"
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
                if uncmax_tmp > self.unc_convergence:
                    self.message_system(
                        "The uncertainty is too large to "
                        "use the last structure."
                    )
                    use_tmp = False
        # Check fmax is lower than previous structure
        if use_tmp and (self.check_fmax or self.check_energy):
            self.update_method(self.best_structures)
            energy_best, fmax_best = self.get_predictions()[1:]
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
            self.copy_best_structures()
            self.message_system("The last structure is used.")
        # Set the best structures as the initial structures for the method
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
            fmax = np.max(self.method.get_fmax())
        return uncmax, energy, fmax

    def get_candidate_predictions(self, **kwargs):
        """
        Get the energies, uncertainties, and fmaxs with the ML calculator
        for the candidates.
        """
        properties = ["fmax", "uncertainty", "energy"]
        results = self.method.get_properties(
            properties=properties,
            allow_calculation=True,
            per_candidate=True,
            **kwargs,
        )
        energies = np.array(results["energy"]).reshape(-1)
        uncertainties = np.array(results["uncertainty"]).reshape(-1)
        fmaxs = np.array(results["fmax"]).reshape(-1)
        return energies, uncertainties, fmaxs

    def parallel_setup(self, comm, **kwargs):
        "Setup the parallelization."
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        return self

    def add_training(self, atoms_list, **kwargs):
        "Add atoms_list data to ML model on rank=0."
        self.mlcalc.add_training(atoms_list)
        return self.mlcalc

    def train_mlmodel(self, point_interest=None, **kwargs):
        "Train the ML model"
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
        return self.mlcalc

    def save_data(self, **kwargs):
        "Save the training data to a file."
        self.mlcalc.save_data(trajectory=self.trainingset)
        return self

    def save_trajectory(self, trajectory, structures, mode="w", **kwargs):
        "Save the trajectory of the data."
        if trajectory is None:
            return self
        if isinstance(trajectory, str):
            with TrajectoryWriter(trajectory, mode=mode) as traj:
                if not isinstance(structures, list):
                    structures = [structures]
                for struc in structures:
                    if self.save_properties_traj:
                        if hasattr(struc.calc, "results"):
                            struc.info["results"] = struc.calc.results
                    traj.write(struc)
        elif isinstance(trajectory, TrajectoryWriter):
            if not isinstance(structures, list):
                structures = [structures]
            for struc in structures:
                if self.save_properties_traj:
                    if hasattr(struc.calc, "results"):
                        struc.info["results"] = struc.calc.results
                trajectory.write(struc)
        else:
            self.message_system(
                "The trajectory type is not supported. "
                "The trajectory is not saved!"
            )
        return self

    def evaluate_candidates(self, candidates, **kwargs):
        "Evaluate the candidates."
        # Check if the candidates are a list
        if not isinstance(candidates, (list, np.ndarray)):
            candidates = [candidates]
        # Evaluate the candidates
        for candidate in candidates:
            # Broadcast the predictions
            self.broadcast_predictions()
            # Evaluate the candidate
            self.evaluate(candidate)
        return self

    def evaluate(self, candidate, **kwargs):
        "Evaluate the ASE atoms with the ASE calculator."
        # Ensure that the candidate is not already in the database
        if self.use_database_check:
            candidate = self.ensure_not_in_database(candidate)
        # Update the evaluated candidate
        self.update_candidate(candidate)
        # Calculate the energies and forces
        self.message_system("Performing evaluation.", end="\r")
        forces = self.candidate.get_forces(
            apply_constraint=self.apply_constraint
        )
        self.energy_true = self.candidate.get_potential_energy(
            force_consistent=self.force_consistent
        )
        self.e_dev = abs(self.energy_true - self.energy_pred)
        self.steps += 1
        self.message_system("Single-point calculation finished.")
        # Store the data
        self.true_fmax = np.nanmax(np.linalg.norm(forces, axis=1))
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
        if np.linalg.norm(cell_old - cell_new) > dtol:
            self.candidate.set_cell(cell_new)
        # Set pbc
        pbc_old = self.candidate.get_pbc()
        pbc_new = candidate.get_pbc()
        if (pbc_old == pbc_new).all():
            self.candidate.set_pbc(pbc_new)
        # Set initial charges
        ini_charge_old = self.candidate.get_initial_charges()
        ini_charge_new = candidate.get_initial_charges()
        if np.linalg.norm(ini_charge_old - ini_charge_new) > dtol:
            self.candidate.set_initial_charges(ini_charge_new)
        # Set initial magmoms
        ini_magmom_old = self.candidate.get_initial_magnetic_moments()
        ini_magmom_new = candidate.get_initial_magnetic_moments()
        if np.linalg.norm(ini_magmom_old - ini_magmom_new) > dtol:
            self.candidate.set_initial_magnetic_moments(ini_magmom_new)
        # Set momenta
        momenta_old = self.candidate.get_momenta()
        momenta_new = candidate.get_momenta()
        if np.linalg.norm(momenta_old - momenta_new) > dtol:
            self.candidate.set_momenta(momenta_new)
        # Set velocities
        velocities_old = self.candidate.get_velocities()
        velocities_new = candidate.get_velocities()
        if np.linalg.norm(velocities_old - velocities_new) > dtol:
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
        # Check if the training set is empty
        if self.get_training_set_size() >= 1:
            return self
        # Calculate the initial structure
        self.evaluate(self.get_structures(get_all=False))
        # Print summary table
        self.print_statement()
        return self

    def update_database_arguments(self, point_interest=None, **kwargs):
        "Update the arguments in the database."
        self.mlcalc.update_database_arguments(
            point_interest=point_interest,
            **kwargs,
        )
        return self

    def ensure_not_in_database(self, atoms, perturb=0.01, **kwargs):
        "Ensure the ASE Atoms object is not in database by perturb it."
        # Return atoms if it does not exist
        if atoms is None:
            return atoms
        # Check if atoms object is in the database
        if self.is_in_database(atoms, **kwargs):
            # Get positions
            pos = atoms.get_positions()
            # Rattle the positions
            pos += np.random.uniform(
                low=-perturb,
                high=perturb,
                size=pos.shape,
            )
            atoms.set_positions(pos)
            self.message_system(
                "The system is rattled, since it is already in the database."
            )
        return atoms

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
        # Get the energies and uncertainties
        energies, uncertainties, fmaxs = self.get_candidate_predictions()
        # Store the uncertainty predictions
        self.umax = np.max(uncertainties)
        self.umean = np.mean(uncertainties)
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
        candidates = self.get_candidates()
        candidates = [candidates[i].copy() for i in i_cand]
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
            # Check the convergence
            if converged:
                self.message_system("Optimization is converged.")
                self.copy_best_structures()
        # Broadcast convergence statement if MPI is used
        converged = broadcast(converged, root=0, comm=self.comm)
        return converged

    def copy_best_structures(self):
        "Copy the best atoms."
        self.best_structures = self.get_structures()
        return self.best_structures

    def get_best_structures(self):
        "Get the best atoms."
        return self.best_structures

    def broadcast_best_structures(self):
        "Broadcast the best atoms."
        self.best_structures = broadcast(
            self.best_structures,
            root=0,
            comm=self.comm,
        )
        return self.best_structures

    def copy_atoms(self, atoms):
        "Copy the ASE Atoms instance with calculator."
        return copy_atoms(atoms)

    def get_objective_str(self, **kwargs):
        "Get what the objective is for the active learning."
        if not self.is_minimization:
            return "max"
        return "min"

    def set_verbose(self, verbose, **kwargs):
        "Set verbose of MLModel."
        self.mlcalc.mlmodel.update_arguments(verbose=verbose)
        return self

    def is_in_database(self, atoms, **kwargs):
        "Check if the ASE Atoms is in the database."
        return self.mlcalc.is_in_database(atoms, **kwargs)

    def save_mlcalc(self, filename="mlcalc.pkl", **kwargs):
        """
        Save the ML calculator object to a file.

        Parameters:
            filename : str
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
            copy_mlcalc : bool
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
            raise Exception(
                "Active learner and Optimization method does "
                "not agree whether to run in parallel!"
            )
        return self

    def message_system(self, message, obj=None, end="\n"):
        "Print output once."
        if self.verbose is True:
            if self.rank == 0:
                if obj is None:
                    print(message, end=end)
                else:
                    print(message, obj, end=end)
        return

    def make_hdr_table(self, **kwargs):
        "Make the header of the summary table for the optimization process."
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
        return hdr

    def make_summary_table(self, **kwargs):
        "Make the summary of the optimization process as table."
        now = datetime.datetime.now().strftime("%d %H:%M:%S")
        # Make the row
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
        return msg

    def save_summary_table(self, msg=None, **kwargs):
        "Save the summary table in the .txt file."
        if self.tabletxt is not None:
            with open(self.tabletxt, "w") as thefile:
                if msg is None:
                    msg = "\n".join(self.print_list)
                thefile.writelines(msg)
        return

    def print_statement(self, **kwargs):
        "Print the Global optimization process as a table"
        msg = ""
        if not self.save_memory or self.rank == 0:
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
        try:
            # Test if the restart is possible
            structure = read(self.trajectory, "0")
            assert len(structure) == self.natoms
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
        except Exception:
            self.message_system(
                "Warning: Restart is not possible! "
                "Reinitalizing active learning."
            )
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
            use_database_check=self.use_database_check,
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
            n_evaluations_each=self.n_evaluations_each,
            min_data=self.min_data,
            save_properties_traj=self.save_properties_traj,
            trajectory=self.trajectory,
            trainingset=self.trainingset,
            converged_trajectory=self.converged_trajectory,
            initial_traj=self.initial_traj,
            tabletxt=self.tabletxt,
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
