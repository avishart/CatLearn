from ase.io import read
from ase.parallel import world
from ase.optimize import FIRE
from .adsorption import AdsorptionAL
from ..optimizer import LocalOptimizer


class MLGO(AdsorptionAL):
    """
    An active learner that is used for accelerating global adsorption search
    using simulated annealing and local optimization with an active learning
    approach.
    The adsorbate is optimized on a surface, where the bond-lengths of the
    adsorbate atoms are fixed and the slab atoms are fixed.
    Afterwards, the structure is local optimized with the initial constraints
    applied to the adsorbate atoms and the surface atoms.
    """

    def __init__(
        self,
        slab,
        adsorbate,
        ase_calc,
        mlcalc=None,
        mlcalc_local=None,
        adsorbate2=None,
        bounds=None,
        opt_kwargs={},
        bond_tol=1e-8,
        chains=None,
        local_opt=FIRE,
        local_opt_kwargs={},
        reuse_data_local=False,
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
        min_data=3,
        use_database_check=True,
        data_perturb=0.001,
        data_tol=1e-8,
        save_properties_traj=True,
        to_save_mlcalc=False,
        save_mlcalc_kwargs={},
        default_mlcalc_kwargs={},
        default_mlcalc_local_kwargs={},
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
        An active learner that is used for accelerating local optimization
        of an atomic structure with an active learning approach.

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
            mlcalc_local: ML-calculator instance.
                The ML-calculator instance used for the local optimization.
                The default BOCalculator instance is used
                if mlcalc_local is None.
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
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            reuse_data_local: bool
                Whether to reuse the data from the global optimization in the
                ML-calculator for the local optimization.
            acq: Acquisition class instance.
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
                The scaling of the fmax convergence criteria.
                It makes the structure(s) converge tighter on surrogate
                surface.
            use_fmax_convergence: bool
                Whether to use the maximum force as an convergence criterion.
            unc_convergence: float
                Maximum uncertainty for convergence in
                the active learning (in eV).
            use_method_unc_conv: bool
                Whether to use the unc_convergence as a convergence criterion
                in the optimization method.
            use_restart: bool
                Use the result from last robust iteration in
                the local optimization.
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
                The number of evaluations for each structure.
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
            default_mlcalc_local_kwargs: dict
                The default keyword arguments for the local ML calculator.
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
        # Save bool for reusing data in the mlcalc_local
        self.reuse_data_local = reuse_data_local
        # Save the local ML-calculator
        self.mlcalc_local = mlcalc_local
        self.default_mlcalc_local_kwargs = default_mlcalc_local_kwargs
        # Initialize the AdsorptionBO
        super().__init__(
            slab=slab,
            adsorbate=adsorbate,
            ase_calc=ase_calc,
            mlcalc=mlcalc,
            adsorbate2=adsorbate2,
            bounds=bounds,
            opt_kwargs=opt_kwargs,
            bond_tol=bond_tol,
            chains=chains,
            acq=acq,
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
            tabletxt=tabletxt,
            timetxt=timetxt,
            prev_calculations=None,
            restart=False,
            seed=seed,
            dtype=dtype,
            comm=comm,
            **kwargs,
        )
        # Get the atomic structure
        atoms = self.get_structures(get_all=False, allow_calculation=False)
        # Build the local method
        self.build_local_method(
            atoms=atoms,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            use_restart=use_restart,
        )
        # Restart the active learning
        prev_calculations = self.restart_optimization(
            restart,
            prev_calculations,
        )
        # Use previous calculations to train ML calculator
        self.use_prev_calculations(prev_calculations)

    def build_local_method(
        self,
        atoms,
        local_opt=FIRE,
        local_opt_kwargs={},
        use_restart=True,
        **kwargs,
    ):
        "Build the local optimization method."
        # Save the instances for creating the local optimizer
        self.atoms = self.copy_atoms(atoms)
        self.local_opt = local_opt
        self.local_opt_kwargs = local_opt_kwargs
        # Set whether to use the restart in the local optimization
        self.use_local_restart = use_restart
        # Build the local optimizer method
        self.local_method = LocalOptimizer(
            atoms,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=False,
            comm=self.comm,
            verbose=self.verbose,
            seed=self.seed,
        )
        return self.local_method

    def setup_mlcalc_local(
        self,
        *args,
        **kwargs,
    ):
        return super(AdsorptionAL, self).setup_mlcalc(*args, **kwargs)

    def run(
        self,
        fmax=0.05,
        steps=200,
        ml_steps=4000,
        ml_steps_local=1000,
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
            ml_steps_local: int
                Maximum number of steps for the local optimization method.
            max_unc: float (optional)
                Maximum uncertainty for continuation of the optimization.
                If max_unc is None, then the optimization is performed
                without the maximum uncertainty.
            dtrust: float (optional)
                The trust distance for the optimization method.
            seed: int (optional)
                The random seed.

        Returns:
            converged: bool
                Whether the active learning is converged.
        """
        # Check if the global optimization is used
        if self.is_global:
            # Run the active learning
            super().run(
                fmax=fmax,
                steps=steps,
                ml_steps=ml_steps,
                max_unc=max_unc,
                dtrust=dtrust,
                **kwargs,
            )
            # Check if the adsorption active learning is converged
            if not self.converged():
                return self.converged()
            # Get the data from the active learning
            data = self.get_data_atoms()
            # Switch to the local optimization
            self.switch_to_local(data)
            # Adjust the number of steps
            steps = steps - self.get_number_of_steps()
            if steps <= 0:
                return self.converged()
        # Run the local active learning
        super().run(
            fmax=fmax,
            steps=steps,
            ml_steps=ml_steps_local,
            max_unc=max_unc,
            dtrust=dtrust,
            **kwargs,
        )
        return self.converged()

    def switch_mlcalcs(self, data, **kwargs):
        """
        Switch the ML calculator used for the local optimization.
        The data is reused, but without the constraints from Adsorption.
        """
        # Get the structures
        structures = self.get_structures(
            get_all=False,
            allow_calculation=False,
        )
        # Setup the ML-calculator for the local optimization
        self.setup_mlcalc_local(
            mlcalc=self.mlcalc_local,
            save_memory=self.save_memory,
            atoms=structures,
            reuse_mlcalc_data=False,
            verbose=self.verbose,
            **self.default_mlcalc_local_kwargs,
        )
        # Add the training data to the local ML-calculator
        self.use_prev_calculations(data)
        return self

    def switch_to_local(self, data, **kwargs):
        "Switch to the local optimization."
        # Reset convergence
        self._converged = False
        # Set the global optimization flag
        self.is_global = False
        # Switch to the local ML-calculator
        self.switch_mlcalcs(data)
        # Store the last structures
        self.structures = self.get_structures(
            get_all=False,
            allow_calculation=False,
        )
        # Use the last structures for the local optimization
        self.local_method.update_optimizable(self.structures)
        # Switch to the local optimization
        self.setup_method(self.local_method)
        # Set whether to use the restart
        self.use_restart = self.use_local_restart
        return self

    def rm_constraints(self, structure, data, **kwargs):
        """
        Remove the constraints from the atoms in the database.
        This is used for the local optimization.
        """
        # Get the constraints from the structures
        constraints = self.get_constraints(structure)
        # Remove the constraints
        for atoms in data:
            atoms.set_constraint(constraints)
        return data

    def build_method(self, *args, **kwargs):
        # Set the global flag to True
        self.is_global = True
        # Build the method for the global optimization
        return super().build_method(*args, **kwargs)

    def use_prev_calculations(self, prev_calculations=None, **kwargs):
        if prev_calculations is None:
            return self
        if isinstance(prev_calculations, str):
            prev_calculations = read(prev_calculations, ":")
        if isinstance(prev_calculations, list) and len(prev_calculations) == 0:
            return self
        # Get the constraints indices if necessary
        if self.is_global or not self.reuse_data_local:
            # Get the constraints of the first calculation
            constraints0 = self.get_constraints_indices(prev_calculations[0])
            # Compare the constraints of the previous calculations
            bool_constraints = [
                self.get_constraints_indices(atoms) == constraints0
                for atoms in prev_calculations[1:]
            ]
        # Check if the prev calculations has the same constraints
        if self.is_global:
            # Check if all constraints are the same
            if not all(bool_constraints):
                self.message_system(
                    "The previous calculations have different constraints. "
                    "Local optimization will be performed."
                )
                # Switch to the local optimization
                self.switch_to_local(prev_calculations)
                return self
        else:
            # Check whether to truncate the previous calculations
            if not self.reuse_data_local:
                # Check if the constraints are different
                if False in bool_constraints:
                    index_local = bool_constraints.index(False)
                    prev_calculations = prev_calculations[index_local:]
                else:
                    # Use only the last calculation
                    prev_calculations = prev_calculations[-1:]
            # Remove the constraints from the previous calculations
            prev_calculations = self.rm_constraints(
                self.get_structures(get_all=False, allow_calculation=False),
                prev_calculations,
            )
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            slab=self.slab,
            adsorbate=self.adsorbate,
            ase_calc=self.ase_calc,
            mlcalc=self.mlcalc,
            mlcalc_local=self.mlcalc_local,
            adsorbate2=self.adsorbate2,
            bounds=self.bounds,
            opt_kwargs=self.opt_kwargs,
            bond_tol=self.bond_tol,
            chains=self.chains,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            reuse_data_local=self.reuse_data_local,
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
            use_restart=self.use_local_restart,
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
            default_mlcalc_local_kwargs=self.default_mlcalc_local_kwargs,
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
        constant_kwargs = dict(is_global=self.is_global)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
