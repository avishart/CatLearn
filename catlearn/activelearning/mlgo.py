from ase.parallel import world
from ase.optimize import FIRE
from .adsorption import AdsorptionAL
from ..optimizer import LocalOptimizer


class MLGO(AdsorptionAL):
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
        chains=None,
        local_opt=FIRE,
        local_opt_kwargs={},
        reuse_data_local=True,
        acq=None,
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
        tabletxt="ml_summary.txt",
        prev_calculations=None,
        restart=False,
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
            n_evaluations_each: int
                The number of evaluations for each structure.
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
                Trajectory filename to store the converged structures.
                Or the TrajectoryWriter instance to store the converged
                structures.
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
        # Initialize the AdsorptionBO
        super().__init__(
            slab=slab,
            adsorbate=adsorbate,
            ase_calc=ase_calc,
            mlcalc=mlcalc,
            adsorbate2=adsorbate2,
            bounds=bounds,
            opt_kwargs=opt_kwargs,
            chains=chains,
            acq=acq,
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
            check_unc=check_unc,
            check_energy=check_energy,
            check_fmax=check_fmax,
            n_evaluations_each=n_evaluations_each,
            min_data=min_data,
            save_properties_traj=save_properties_traj,
            trajectory=trajectory,
            trainingset=trainingset,
            converged_trajectory=converged_trajectory,
            tabletxt=tabletxt,
            prev_calculations=prev_calculations,
            restart=restart,
            comm=comm,
            **kwargs,
        )
        # Get the atomic structure
        atoms = self.get_structures(get_all=False)
        # Build the local method
        self.build_local_method(
            atoms=atoms,
            mlcalc_local=mlcalc_local,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            reuse_data_local=reuse_data_local,
            use_restart=use_restart,
        )
        # Save the local ML-calculator
        self.mlcalc_local = mlcalc_local

    def build_local_method(
        self,
        atoms,
        local_opt=FIRE,
        local_opt_kwargs={},
        reuse_data_local=True,
        use_restart=True,
        **kwargs,
    ):
        "Build the local optimization method."
        # Save the instances for creating the local optimizer
        self.atoms = self.copy_atoms(atoms)
        self.local_opt = local_opt
        self.local_opt_kwargs = local_opt_kwargs
        # Save bool for reusing data in the mlcalc_local
        self.reuse_data_local = reuse_data_local
        # Build the local optimizer method
        self.local_method = LocalOptimizer(
            atoms,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=False,
            comm=self.comm,
            verbose=self.verbose,
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
            ml_steps_local: int
                Maximum number of steps for the local optimization method.
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
        # Run the active learning
        super().run(
            fmax=fmax,
            steps=steps,
            ml_steps=ml_steps,
            max_unc=max_unc,
            dtrust=dtrust,
            seed=seed,
            **kwargs,
        )
        # Check if the active learning is converged
        if not self.converged():
            return self.converged()
        # Switch to the local optimization
        self.switch_to_local()
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
            seed=seed,
            **kwargs,
        )
        return self.converged()

    def switch_mlcalcs(self, **kwargs):
        """
        Switch the ML calculator used for the local optimization.
        The data is reused, but without the constraints from Adsorption.
        """
        # Get the data from the active learning
        data = self.get_data_atoms()
        if not self.reuse_data_local:
            data = data[-1:]
        # Get the structures
        structures = self.get_structures(get_all=False)
        # Setup the ML-calculator for the local optimization
        self.setup_mlcalc_local(
            mlcalc_local=self.mlcalc_local,
            save_memory=self.save_memory,
            atoms=structures,
            verbose=self.verbose,
            **kwargs,
        )
        # Remove adsorption constraints
        constraints = [c.copy() for c in structures.constraints]
        for atoms in data:
            atoms.set_constraint(constraints)
        self.use_prev_calculations(data)
        return self

    def switch_to_local(self, **kwargs):
        "Switch to the local optimization."
        # Reset convergence
        self._converged = False
        # Switch to the local ML-calculator
        self.switch_mlcalcs()
        # Store the last structures
        self.structures = self.get_structures()
        # Use the last structures for the local optimization
        self.local_method.update_optimizable(self.structures)
        # Switch to the local optimization
        self.setup_method(self.local_method)
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
            chains=self.chains,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            reuse_data_local=self.reuse_data_local,
            acq=self.acq,
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
            tabletxt=self.tabletxt,
            comm=self.comm,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
