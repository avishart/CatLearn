from ase.parallel import world
from ase.optimize import FIRE
from .activelearning import ActiveLearning
from ..optimizer import RandomAdsorptionOptimizer
from ..optimizer import ParallelOptimizer
from ..regression.gp.baseline import BornRepulsionCalculator, MieCalculator


class RandomAdsorptionAL(ActiveLearning):
    """
    An active learner that is used for accelerating global adsorption search
    using random sampling and local optimization with an active learning
    approach.
    The adsorbate is random sampled in space and the most stable structure is
    local optimized.
    """

    def __init__(
        self,
        slab,
        adsorbate,
        ase_calc,
        mlcalc=None,
        adsorbate2=None,
        bounds=None,
        n_random_draws=200,
        use_initial_opt=False,
        initial_fmax=0.2,
        initial_steps=50,
        use_repulsive_check=True,
        repulsive_tol=0.1,
        repulsive_calculator=BornRepulsionCalculator(),
        local_opt=FIRE,
        local_opt_kwargs={},
        chains=None,
        acq=None,
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
        min_data=5,
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
        last_traj=None,
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
            slab: Atoms instance
                The slab structure.
                Can either be a surface or a nanoparticle.
            adsorbate: Atoms instance
                The adsorbate structure.
            ase_calc: ASE calculator instance.
                ASE calculator as implemented in ASE.
            mlcalc: ML-calculator instance.
                The ML-calculator instance used as surrogate surface.
                The default BOCalculator instance is used if mlcalc is None.
            adsorbate2: Atoms instance (optional)
                The second adsorbate structure.
                Optimize both adsorbates simultaneously.
                The two adsorbates will have different tags.
            bounds: (6,2) array or (12,2) array (optional)
                The bounds for the optimization.
                The first 3 rows are the x, y, z scaled coordinates for
                the center of the adsorbate.
                The next 3 rows are the three rotation angles in radians.
                If two adsorbates are optimized, the next 6 rows are for
                the second adsorbate.
            n_random_draws: int
                The number of random structures to be drawn.
                If chains is not None, then the number of random
                structures is n_random_draws * chains.
            use_initial_opt: bool
                If True, the initial structures, drawn from the random
                sampling, will be local optimized before the structure
                with lowest energy are local optimized.
            initial_fmax: float
                The maximum force for the initial local optimizations.
            initial_steps: int
                The maximum number of steps for the initial local
                optimizations.
            use_repulsive_check: bool
                If True, a energy will be calculated for each randomly
                drawn structure to check if the energy is not too large.
            repulsive_tol: float
                The tolerance for the repulsive energy check.
            repulsive_calculator: ASE calculator instance
                The calculator used for the repulsive energy check.
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            chains: int (optional)
                The number of optimization that will be run in parallel.
                It is only used if parallel_run=True.
            acq: Acquisition class instance.
                The Acquisition instance used for calculating the
                acq. function and choose a candidate to calculate next.
                The default AcqUME instance is used if acq is None.
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
                The scaling of the fmax convergence criterion.
                It makes the structure(s) converge tighter on surrogate
                surface.
                If use_database_check is True and the structure is in the
                database, then the scale_fmax is multiplied by the original
                scale_fmax to give tighter convergence.
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
                Be aware that restart and low max_unc can result in only the
                initial structure passing the maximum uncertainty criterion.
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
                Number of evaluations for each candidate.
            min_data: int
                The minimum number of data points in the training set before
                the active learning can converge.
            use_database_check: bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled.
                Please be aware that the predicted structure will differ from
                the structure in the database if the rattling is applied.
                If use_database_check is True and the structure is in the
                database, then the scale_fmax is multiplied by the original
                scale_fmax to give tighter convergence.
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
            converged_trajectory: str or TrajectoryWriter instance (optional)
                Trajectory filename to store the converged structure(s).
                Or the TrajectoryWriter instance to store the converged
                structure(s).
            initial_traj: str or TrajectoryWriter instance (optional)
                Trajectory filename to store the initial structure(s).
                Or the TrajectoryWriter instance to store the initial
                structure(s).
            last_traj: str or TrajectoryWriter instance (optional)
                Trajectory filename to store the last structure(s).
                Or the TrajectoryWriter instance to store the last
                structure(s).
            tabletxt: str
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
        # Build the optimizer method
        method = self.build_method(
            slab=slab,
            adsorbate=adsorbate,
            adsorbate2=adsorbate2,
            bounds=bounds,
            n_random_draws=n_random_draws,
            use_initial_struc=use_restart,
            use_initial_opt=use_initial_opt,
            initial_fmax=initial_fmax,
            initial_steps=initial_steps,
            use_repulsive_check=use_repulsive_check,
            repulsive_tol=repulsive_tol,
            repulsive_calculator=repulsive_calculator,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            chains=chains,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        # Initialize the BayesianOptimizer
        super().__init__(
            method=method,
            ase_calc=ase_calc,
            mlcalc=mlcalc,
            acq=acq,
            is_minimization=True,
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
            use_database_check=use_database_check,
            data_perturb=data_perturb,
            data_tol=data_tol,
            save_properties_traj=save_properties_traj,
            to_save_mlcalc=to_save_mlcalc,
            save_mlcalc_kwargs=save_mlcalc_kwargs,
            default_mlcalc_kwargs=default_mlcalc_kwargs,
            trajectory=trajectory,
            trainingset=trainingset,
            pred_evaluated=pred_evaluated,
            converged_trajectory=converged_trajectory,
            initial_traj=initial_traj,
            last_traj=last_traj,
            tabletxt=tabletxt,
            timetxt=timetxt,
            prev_calculations=prev_calculations,
            restart=restart,
            seed=seed,
            dtype=dtype,
            comm=comm,
            **kwargs,
        )

    def build_method(
        self,
        slab,
        adsorbate,
        adsorbate2=None,
        bounds=None,
        n_random_draws=20,
        use_initial_struc=True,
        use_initial_opt=False,
        initial_fmax=0.2,
        initial_steps=50,
        use_repulsive_check=True,
        repulsive_tol=0.1,
        repulsive_calculator=BornRepulsionCalculator(),
        local_opt=FIRE,
        local_opt_kwargs={},
        chains=None,
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        "Build the optimization method."
        # Save the instances for creating the adsorption optimizer
        self.slab = self.copy_atoms(slab)
        self.adsorbate = self.copy_atoms(adsorbate)
        if adsorbate2 is not None:
            self.adsorbate2 = self.copy_atoms(adsorbate2)
        else:
            self.adsorbate2 = None
        self.bounds = bounds
        self.n_random_draws = n_random_draws
        self.use_initial_struc = use_initial_struc
        self.use_initial_opt = use_initial_opt
        self.initial_fmax = initial_fmax
        self.initial_steps = initial_steps
        self.use_repulsive_check = use_repulsive_check
        self.repulsive_tol = repulsive_tol
        self.repulsive_calculator = repulsive_calculator
        self.local_opt = local_opt
        self.local_opt_kwargs = local_opt_kwargs
        self.chains = chains
        # Build the optimizer method
        method = RandomAdsorptionOptimizer(
            slab=slab,
            adsorbate=adsorbate,
            adsorbate2=adsorbate2,
            bounds=bounds,
            n_random_draws=n_random_draws,
            use_initial_struc=use_initial_struc,
            use_initial_opt=use_initial_opt,
            initial_fmax=initial_fmax,
            initial_steps=initial_steps,
            use_repulsive_check=use_repulsive_check,
            repulsive_tol=repulsive_tol,
            repulsive_calculator=repulsive_calculator,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=False,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        # Run the method in parallel if requested
        if parallel_run:
            method = ParallelOptimizer(
                method,
                chains=chains,
                parallel_run=parallel_run,
                comm=comm,
                verbose=verbose,
                seed=seed,
            )
        return method

    def extra_initial_data(self, **kwargs):
        # Get the number of training data
        n_data = self.get_training_set_size()
        # Check if the training set is empty
        if n_data >= 2:
            return self
        # Get the initial structures from baseline potentials
        method_extra = self.method.copy()
        method_extra.update_arguments(
            n_random_draws=20,
            use_initial_opt=False,
            use_repulsive_check=True,
        )
        if n_data == 0:
            method_extra.set_calculator(BornRepulsionCalculator(r_scale=1.0))
        else:
            method_extra.set_calculator(
                MieCalculator(r_scale=1.2, denergy=0.2)
            )
        method_extra.run(fmax=0.1, steps=21)
        atoms = method_extra.get_candidates()[0]
        # Evaluate the structure
        self.evaluate(atoms)
        # Print summary table
        self.print_statement()
        # Check if another initial data is needed
        if n_data == 0:
            self.extra_initial_data(**kwargs)
        return self

    def setup_default_mlcalc(
        self,
        kappa=-1.0,
        calc_kwargs={},
        **kwargs,
    ):
        # Set a limit for the uncertainty
        if "max_unc" not in calc_kwargs.keys():
            calc_kwargs["max_unc"] = 2.0
        return super().setup_default_mlcalc(
            kappa=kappa,
            calc_kwargs=calc_kwargs,
            **kwargs,
        )

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            slab=self.slab,
            adsorbate=self.adsorbate,
            ase_calc=self.ase_calc,
            mlcalc=self.mlcalc,
            adsorbate2=self.adsorbate2,
            bounds=self.bounds,
            n_random_draws=self.n_random_draws,
            use_initial_opt=self.use_initial_opt,
            initial_fmax=self.initial_fmax,
            initial_steps=self.initial_steps,
            use_repulsive_check=self.use_repulsive_check,
            repulsive_tol=self.repulsive_tol,
            repulsive_calculator=self.repulsive_calculator,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            chains=self.chains,
            acq=self.acq,
            save_memory=self.save_memory,
            parallel_run=self.parallel_run,
            copy_calc=self.copy_calc,
            verbose=self.verbose,
            apply_constraint=self.apply_constraint,
            force_consistent=self.force_consistent,
            scale_fmax=self.scale_fmax_org,
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
            last_traj=self.last_traj,
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
