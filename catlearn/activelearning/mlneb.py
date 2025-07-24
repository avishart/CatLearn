from ase.optimize import FIRE
from ase.parallel import world
from ase.io import read
from .activelearning import ActiveLearning
from ..optimizer import LocalCINEB
from ..structures.neb import ImprovedTangentNEB, OriginalNEB


class MLNEB(ActiveLearning):
    """
    An active learner that is used for accelerating nudged elastic band
    (NEB) optimization with an active learning approach.
    """

    def __init__(
        self,
        start,
        end,
        ase_calc,
        mlcalc=None,
        neb_method=ImprovedTangentNEB,
        neb_kwargs={},
        n_images=15,
        climb=True,
        neb_interpolation="linear",
        neb_interpolation_kwargs={},
        start_without_ci=True,
        reuse_ci_path=True,
        local_opt=FIRE,
        local_opt_kwargs={},
        acq=None,
        save_memory=False,
        parallel_run=False,
        copy_calc=False,
        verbose=True,
        apply_constraint=True,
        force_consistent=False,
        scale_fmax=0.8,
        unc_convergence=0.02,
        use_method_unc_conv=True,
        use_restart=True,
        check_unc=True,
        check_energy=False,
        check_fmax=True,
        max_unc_restart=0.05,
        n_evaluations_each=1,
        min_data=3,
        use_database_check=True,
        data_perturb=0.001,
        data_tol=1e-8,
        save_properties_traj=True,
        to_save_mlcalc=True,
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
            start: Atoms instance or ASE Trajectory file.
                The Atoms must have the calculator attached with energy.
                Initial end-point of the NEB path.
            end: Atoms instance or ASE Trajectory file.
                The Atoms must have the calculator attached with energy.
                Final end-point of the NEB path.
            ase_calc: ASE calculator instance.
                ASE calculator as implemented in ASE.
            mlcalc: ML-calculator instance.
                The ML-calculator instance used as surrogate surface.
                The default BOCalculator instance is used if mlcalc is None.
            neb_method: NEB class object or str
                The NEB implemented class object used for the ML-NEB.
                A string can be used to select:
                - 'improvedtangentneb' (default)
                - 'ewneb'
            neb_kwargs: dict
                A dictionary with the arguments used in the NEB object
                to create the instance.
                Climb must not be included.
            n_images: int
                Number of images of the path (if not included a path before).
                The number of images include the 2 end-points of the NEB path.
            climb: bool
                Whether to use the climbing image in the NEB.
                It is strongly recommended to have climb=True.
            neb_interpolation: str or list of ASE Atoms or ASE Trajectory file
                The interpolation method used to create the NEB path.
                The string can be:
                - 'linear' (default)
                - 'idpp'
                - 'rep'
                - 'born
                - 'ends'
                Otherwise, the premade images can be given as a list of
                ASE Atoms.
                A string of the ASE Trajectory file that contains the images
                can also be given.
            neb_interpolation_kwargs: dict
                The keyword arguments for the interpolation method.
                It is only used when the interpolation method is a string.
            start_without_ci: bool
                Whether to start the NEB without the climbing image.
                If True, the NEB path will be optimized without
                the climbing image and afterwards climbing image is used
                if climb=True as well.
                If False, the NEB path will be optimized with the climbing
                image if climb=True as well.
            reuse_ci_path: bool
                Whether to restart from the climbing image path when the NEB
                without climbing image is converged.
            local_opt: ASE optimizer object
                The local optimizer object.
            local_opt_kwargs: dict
                The keyword arguments for the local optimizer.
            acq: Acquisition class instance.
                The Acquisition instance used for calculating the
                acq. function and choose a candidate to calculate next.
                The default AcqUME instance is used if acq is None.
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
        # Save the end points for creating the NEB
        self.setup_endpoints(start, end, prev_calculations)
        # Build the optimizer method and NEB within
        method = self.build_method(
            neb_method=neb_method,
            neb_kwargs=neb_kwargs,
            climb=climb,
            n_images=n_images,
            neb_interpolation=neb_interpolation,
            neb_interpolation_kwargs=neb_interpolation_kwargs,
            start_without_ci=start_without_ci,
            reuse_ci_path=reuse_ci_path,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )
        # Initialize the BayesianOptimizer
        super().__init__(
            method=method,
            ase_calc=ase_calc,
            mlcalc=mlcalc,
            acq=acq,
            is_minimization=False,
            save_memory=save_memory,
            parallel_run=parallel_run,
            copy_calc=copy_calc,
            verbose=verbose,
            apply_constraint=apply_constraint,
            force_consistent=force_consistent,
            scale_fmax=scale_fmax,
            use_fmax_convergence=climb,
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
            trajectory=trajectory,
            trainingset=trainingset,
            pred_evaluated=pred_evaluated,
            converged_trajectory=converged_trajectory,
            initial_traj=initial_traj,
            tabletxt=tabletxt,
            timetxt=timetxt,
            prev_calculations=self.prev_calculations,
            restart=restart,
            seed=seed,
            dtype=dtype,
            comm=comm,
            **kwargs,
        )

    def setup_endpoints(
        self,
        start,
        end,
        prev_calculations,
        tol=1e-8,
        **kwargs,
    ):
        """
        Setup the start and end points for the NEB calculation.
        """
        # Load the start and end points from trajectory files
        if isinstance(start, str):
            start = read(start)
        if isinstance(end, str):
            end = read(end)
        # Save the start point with calculators
        try:
            start.get_forces()
        except RuntimeError:
            raise RuntimeError(
                "The start point must have a calculator attached with "
                "energy and forces!"
            )
        self.start = self.copy_atoms(start)
        # Save the end point with calculators
        try:
            end.get_forces()
        except RuntimeError:
            raise RuntimeError(
                "The end point must have a calculator attached with "
                "energy and forces!"
            )
        self.end = self.copy_atoms(end)
        # Save in previous calculations
        self.prev_calculations = [self.start, self.end]
        if prev_calculations is not None:
            if isinstance(prev_calculations, str):
                prev_calculations = read(prev_calculations, ":")
            # Check if end points are in the previous calculations
            if len(prev_calculations):
                is_same = self.compare_atoms(
                    self.start,
                    prev_calculations[0],
                    tol=tol,
                )
                if is_same:
                    prev_calculations = prev_calculations[1:]
            if len(prev_calculations):
                is_same = self.compare_atoms(
                    self.end,
                    prev_calculations[0],
                    tol=tol,
                )
                if is_same:
                    prev_calculations = prev_calculations[1:]
            # Save the previous calculations
            self.prev_calculations += list(prev_calculations)
        return self

    def build_method(
        self,
        neb_method,
        neb_kwargs={},
        climb=True,
        n_images=15,
        k=3.0,
        remove_rotation_and_translation=False,
        mic=True,
        neb_interpolation="linear",
        neb_interpolation_kwargs={},
        start_without_ci=True,
        reuse_ci_path=True,
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        "Build the optimization method."
        # Save the instances for creating the local optimizer
        self.local_opt = local_opt
        self.local_opt_kwargs = local_opt_kwargs
        # Save the instances for creating the NEB
        self.neb_method = neb_method
        self.neb_kwargs = dict(
            k=k,
            remove_rotation_and_translation=remove_rotation_and_translation,
            parallel=parallel_run,
        )
        if isinstance(neb_method, str) or issubclass(neb_method, OriginalNEB):
            self.neb_kwargs.update(
                dict(
                    use_image_permutation=False,
                    save_properties=True,
                    mic=mic,
                    comm=comm,
                )
            )
        else:
            self.neb_kwargs.update(dict(world=comm))
        self.neb_kwargs.update(neb_kwargs)
        self.n_images = n_images
        self.neb_interpolation = neb_interpolation
        self.neb_interpolation_kwargs = dict(
            mic=mic,
            remove_rotation_and_translation=remove_rotation_and_translation,
        )
        self.neb_interpolation_kwargs.update(neb_interpolation_kwargs)
        self.start_without_ci = start_without_ci
        self.climb = climb
        self.reuse_ci_path = reuse_ci_path
        # Build the sequential neb optimizer
        method = LocalCINEB(
            start=self.start,
            end=self.end,
            neb_method=self.neb_method,
            neb_kwargs=self.neb_kwargs,
            n_images=self.n_images,
            climb=self.climb,
            neb_interpolation=self.neb_interpolation,
            neb_interpolation_kwargs=self.neb_interpolation_kwargs,
            start_without_ci=self.start_without_ci,
            reuse_ci_path=self.reuse_ci_path,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        return method

    def extra_initial_data(self, **kwargs):
        # Check if the training set is empty
        if self.get_training_set_size() >= 3:
            return self
        # Get the images
        images = self.get_structures(get_all=True, allow_calculation=False)
        # Calculate energies of end points
        e_start = self.start.get_potential_energy()
        e_end = self.end.get_potential_energy()
        # Get the image with the potential highest energy
        if e_start >= e_end:
            i_middle = int((len(images) - 2) / 3.0)
        else:
            i_middle = int(2.0 * (len(images) - 2) / 3.0)
        candidate = images[1 + i_middle].copy()
        # Calculate the structure
        self.evaluate(candidate)
        # Print summary table
        self.print_statement()
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            start=self.start,
            end=self.end,
            ase_calc=self.ase_calc,
            mlcalc=self.mlcalc,
            neb_method=self.neb_method,
            neb_kwargs=self.neb_kwargs,
            n_images=self.n_images,
            climb=self.climb,
            neb_interpolation=self.neb_interpolation,
            neb_interpolation_kwargs=self.neb_interpolation_kwargs,
            start_without_ci=self.start_without_ci,
            reuse_ci_path=self.reuse_ci_path,
            local_opt=self.local_opt,
            local_opt_kwargs=self.local_opt_kwargs,
            acq=self.acq,
            is_minimization=self.is_minimization,
            save_memory=self.save_memory,
            parallel_run=self.parallel_run,
            copy_calc=self.copy_calc,
            verbose=self.verbose,
            apply_constraint=self.apply_constraint,
            force_consistent=self.force_consistent,
            scale_fmax=self.scale_fmax,
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
