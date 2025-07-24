from ase.parallel import world
from .activelearning import ActiveLearning
from ..optimizer import AdsorptionOptimizer
from ..optimizer import ParallelOptimizer
from ..regression.gp.baseline import BornRepulsionCalculator, MieCalculator


class AdsorptionAL(ActiveLearning):
    """
    An active learner that is used for accelerating global adsorption search
    using simulated annealing with an active learning approach.
    The adsorbate is optimized on a surface, where the bond-lengths of the
    adsorbate atoms are fixed and the slab atoms are fixed.
    """

    def __init__(
        self,
        slab,
        adsorbate,
        ase_calc,
        mlcalc=None,
        adsorbate2=None,
        bounds=None,
        opt_kwargs={},
        bond_tol=1e-8,
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
        n_evaluations_each=1,
        min_data=5,
        use_database_check=True,
        data_perturb=0.001,
        data_tol=1e-8,
        save_properties_traj=True,
        to_save_mlcalc=False,
        save_mlcalc_kwargs={},
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
            opt_kwargs: dict
                The keyword arguments for the simulated annealing optimizer.
            bond_tol: float
                The bond tolerance used for the FixBondLengths.
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
            opt_kwargs=opt_kwargs,
            bond_tol=bond_tol,
            chains=chains,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
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
            use_restart=False,
            n_evaluations_each=n_evaluations_each,
            min_data=min_data,
            use_database_check=use_database_check,
            data_perturb=data_perturb,
            data_tol=data_tol,
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
        opt_kwargs={},
        bond_tol=1e-8,
        chains=None,
        parallel_run=False,
        comm=world,
        verbose=False,
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
        self.opt_kwargs = opt_kwargs.copy()
        self.bond_tol = bond_tol
        self.chains = chains
        # Build the optimizer method
        method = AdsorptionOptimizer(
            slab=slab,
            adsorbate=adsorbate,
            adsorbate2=adsorbate2,
            bounds=bounds,
            opt_kwargs=opt_kwargs,
            bond_tol=bond_tol,
            parallel_run=False,
            comm=comm,
            verbose=verbose,
        )
        # Run the method in parallel if requested
        if parallel_run:
            method = ParallelOptimizer(
                method,
                chains=chains,
                parallel_run=parallel_run,
                comm=comm,
                verbose=verbose,
            )
        return method

    def extra_initial_data(self, **kwargs):
        # Get the number of training data
        n_data = self.get_training_set_size()
        # Check if the training set is empty
        if n_data >= 2:
            return self
        # Get the initial structures from baseline potentials
        if n_data == 0:
            self.method.set_calculator(BornRepulsionCalculator(r_scale=1.0))
        else:
            self.method.set_calculator(MieCalculator(r_scale=1.2, denergy=0.2))
        self.method.run(fmax=0.05, steps=1000)
        atoms = self.method.get_candidates()[0]
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
        fp=None,
        atoms=None,
        baseline=BornRepulsionCalculator(),
        use_derivatives=True,
        calc_forces=False,
        kappa=-3.0,
        **kwargs,
    ):
        from ..regression.gp.fingerprint import SortedInvDistances

        # Setup the fingerprint
        if fp is None:
            # Check if the Atoms object is given
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
                fp = SortedInvDistances(
                    reduce_dimensions=True,
                    use_derivatives=True,
                    periodic_softmax=periodic_softmax,
                    wrap=True,
                    use_tags=True,
                )
        return super().setup_default_mlcalc(
            fp=fp,
            atoms=atoms,
            baseline=baseline,
            use_derivatives=use_derivatives,
            calc_forces=calc_forces,
            kappa=kappa,
            **kwargs,
        )

    def get_constraints(self, structure, **kwargs):
        "Get the constraints of the structures in the method."
        constraints = [c.copy() for c in structure.constraints]
        return constraints

    def get_constraints_indices(self, structure, **kwargs):
        "Get the indices of the constraints of the structures in the method."
        indices = [i for c in structure.constraints for i in c.get_indices()]
        return indices

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
            opt_kwargs=self.opt_kwargs,
            bond_tol=self.bond_tol,
            chains=self.chains,
            acq=self.acq,
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
