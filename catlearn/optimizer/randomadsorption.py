from .local import LocalOptimizer
from ase.parallel import world
from ase.optimize import FIRE
from numpy import array, asarray, concatenate, cos, inf, matmul, pi, sin
from ..regression.gp.baseline import BornRepulsionCalculator


class RandomAdsorptionOptimizer(LocalOptimizer):
    """
    The RandomAdsorptionOptimizer is used to run a global optimization of
    an adsorption on a surface.
    A single structure will be created and optimized.
    Random structures will be sampled and the most stable structure is local
    optimized.
    The RandomAdsorptionOptimizer is applicable to be used with
    active learning.
    """

    def __init__(
        self,
        slab,
        adsorbate,
        adsorbate2=None,
        bounds=None,
        n_random_draws=50,
        use_initial_struc=True,
        use_initial_opt=False,
        initial_fmax=0.2,
        initial_steps=50,
        use_repulsive_check=True,
        repulsive_tol=0.1,
        repulsive_calculator=BornRepulsionCalculator(),
        local_opt=FIRE,
        local_opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        seed=None,
        **kwargs,
    ):
        """
        Initialize the OptimizerMethod instance.

        Parameters:
            slab: Atoms instance
                The slab structure.
            adsorbate: Atoms instance
                The adsorbate structure.
            adsorbate2: Atoms instance (optional)
                The second adsorbate structure.
            bounds: (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for drawing the positions
                for the adsorbate(s).
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.
            n_random_draws: int
                The number of random structures to be drawn.
            use_initial_struc: bool
                If True, the initial structure is used as one of the drawn
                structures.
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
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
            seed: int (optional)
                The random seed for the optimization.
                The seed an also be a RandomState or Generator instance.
                If not given, the default random number generator is used.
        """
        # Set the verbose
        self.verbose = verbose
        # Create the atoms object from the slab and adsorbate
        self.create_slab_ads(slab, adsorbate, adsorbate2)
        # Create the boundary conditions
        self.setup_bounds(bounds)
        # Set the parameters
        self.update_arguments(
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
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )
        # Make initial optimizable structure
        self.make_initial_structure()

    def create_slab_ads(
        self,
        slab,
        adsorbate,
        adsorbate2=None,
        **kwargs,
    ):
        """
        Create the structure for the adsorption optimization.

        Parameters:
            slab: Atoms object
                The slab structure.
            adsorbate: Atoms object
                The adsorbate structure.
            adsorbate2: Atoms object (optional)
                The second adsorbate structure.

        Returns:
            self: object
                The object itself.
        """
        # Check the slab and adsorbate are given
        if slab is None or adsorbate is None:
            raise ValueError("The slab and adsorbate must be given!")
        # Setup the slab
        self.n_slab = len(slab)
        self.slab = slab.copy()
        self.slab.set_tags(0)
        optimizable = self.slab.copy()
        # Setup the adsorbate
        self.n_ads = len(adsorbate)
        self.adsorbate = adsorbate.copy()
        self.adsorbate.set_tags(1)
        self.adsorbate.cell = optimizable.cell.copy()
        self.adsorbate.pbc = optimizable.pbc.copy()
        pos_ads = self.adsorbate.get_positions()
        pos_ads -= pos_ads.mean(axis=0)
        self.adsorbate.set_positions(pos_ads)
        optimizable.extend(self.adsorbate.copy())
        # Setup the adsorbate2
        if adsorbate2 is not None:
            self.n_ads2 = len(adsorbate2)
            self.adsorbate2 = adsorbate2.copy()
            self.adsorbate2.set_tags(2)
            self.adsorbate2.cell = optimizable.cell.copy()
            self.adsorbate2.pbc = optimizable.pbc.copy()
            pos_ads2 = self.adsorbate2.get_positions()
            pos_ads2 -= pos_ads2.mean(axis=0)
            self.adsorbate2.set_positions(pos_ads2)
            optimizable.extend(self.adsorbate2.copy())
        else:
            self.n_ads2 = 0
            self.adsorbate2 = None
        # Get the full number of atoms
        self.natoms = len(optimizable)
        # Store the positions and cell
        self.positions0 = optimizable.get_positions().copy()
        self.cell = array(optimizable.get_cell())
        # Setup the optimizable structure
        self.setup_optimizable(optimizable)
        return self

    def setup_bounds(self, bounds=None):
        """
        Setup the boundary conditions for the global optimization.

        Parameters:
            bounds: (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for drawing the positions
                for the adsorbate(s).
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.

        Returns:
            self: object
                The object itself.
        """
        # Check the bounds are given
        if bounds is None:
            # Make default bounds
            self.bounds = asarray(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 2.0 * pi],
                    [0.0, 2.0 * pi],
                    [0.0, 2.0 * pi],
                ]
            )
        else:
            self.bounds = bounds.copy()
        # Check the bounds have the correct shape
        if self.n_ads2 == 0 and self.bounds.shape != (6, 2):
            raise ValueError("The bounds must have shape (6,2)!")
        elif self.n_ads2 > 0 and not (
            self.bounds.shape == (6, 2) or self.bounds.shape == (12, 2)
        ):
            raise ValueError("The bounds must have shape (6,2) or (12,2)!")
        # Check if the bounds are for two adsorbates
        if self.n_ads2 > 0 and self.bounds.shape[0] == 6:
            self.bounds = concatenate([self.bounds, self.bounds], axis=0)
        return self

    def run(
        self,
        fmax=0.05,
        steps=1000000,
        max_unc=None,
        dtrust=None,
        unc_convergence=None,
        **kwargs,
    ):
        # Check if the optimization can take any steps
        if steps <= 0:
            return self._converged
        # Take initial structure into account
        n_random_draws = self.n_random_draws
        if self.use_initial_struc:
            n_random_draws -= 1
        # Draw random structures
        x_drawn = self.draw_random_structures(
            n_random_draws=n_random_draws,
            **kwargs,
        )
        # Get the best drawn structure
        best_pos, steps = self.get_best_drawn_structure(
            x_drawn,
            steps=steps,
            max_unc=max_unc,
            dtrust=dtrust,
            **kwargs,
        )
        # Set the positions
        self.optimizable.set_positions(best_pos)
        # Check if the optimization can take any steps
        if steps <= 0:
            self.message(
                "No steps left after drawing random structures.",
                is_warning=True,
            )
            return self._converged
        # Run the local optimization
        converged, _ = self.local_optimize(
            atoms=self.optimizable,
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

    def is_parallel_allowed(self):
        return False

    def update_arguments(
        self,
        slab=None,
        adsorbate=None,
        adsorbate2=None,
        bounds=None,
        n_random_draws=None,
        use_initial_struc=None,
        use_initial_opt=None,
        initial_fmax=None,
        initial_steps=None,
        use_repulsive_check=None,
        repulsive_tol=None,
        repulsive_calculator=None,
        local_opt=None,
        local_opt_kwargs=None,
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
            slab: Atoms instance
                The slab structure.
            adsorbate: Atoms instance
                The adsorbate structure.
            adsorbate2: Atoms instance (optional)
                The second adsorbate structure.
            bounds: (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for drawing the positions
                for the adsorbate(s).
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.
            n_random_draws: int
                The number of random structures to be drawn.
            use_initial_struc: bool
                If True, the initial structure is used as one of the drawn
                structures.
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
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.
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
            optimizable=None,
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
        )
        # Create the atoms object from the slab and adsorbate
        if slab is not None or adsorbate is not None or adsorbate2 is not None:
            if slab is None:
                slab = self.slab.copy()
            if adsorbate is None:
                adsorbate = self.adsorbate.copy()
            if adsorbate2 is None and self.adsorbate2 is not None:
                adsorbate2 = self.adsorbate2.copy()
            self.create_slab_ads(
                slab,
                adsorbate,
                adsorbate2,
            )
        # Create the boundary conditions
        if bounds is not None:
            self.setup_bounds(bounds)
        # Set the rest of the parameters
        if n_random_draws is not None:
            self.n_random_draws = int(n_random_draws)
        if use_initial_struc is not None:
            self.use_initial_struc = use_initial_struc
        if use_initial_opt is not None:
            self.use_initial_opt = use_initial_opt
        if initial_fmax is not None:
            self.initial_fmax = float(initial_fmax)
        if initial_steps is not None:
            self.initial_steps = int(initial_steps)
        if use_repulsive_check is not None:
            self.use_repulsive_check = use_repulsive_check
        if repulsive_tol is not None:
            self.repulsive_tol = float(repulsive_tol)
        if repulsive_calculator is not None or not hasattr(
            self, "repulsive_calculator"
        ):
            self.repulsive_calculator = repulsive_calculator
        return self

    def draw_random_structures(self, n_random_draws=50, **kwargs):
        "Draw random structures for the adsorption optimization."
        # Get reference energy
        self.e_ref = self.get_reference_energy()
        # Initialize the drawn structures
        failed_steps = 0
        n_drawn = 0
        x_drawn = []
        # Set a dummy structure for the repulsive check
        if self.use_repulsive_check:
            dummy_optimizable = self.optimizable.copy()
            dummy_optimizable.calc = self.repulsive_calculator
        # Draw random structures
        while n_drawn < n_random_draws:
            # Draw a random structure
            x = self.rng.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1])
            # Evaluate the value of the structure
            if self.use_repulsive_check:
                e = self.evaluate_value(x, atoms=dummy_optimizable)
                # Check if the value is not too large
                if e - self.e_ref > self.repulsive_tol:
                    failed_steps += 1
                    if failed_steps > 100.0 * n_random_draws:
                        self.message(
                            f"{failed_steps} failed drawns. "
                            "Stopping is recommended!",
                            is_warning=True,
                        )
                    continue
            # Add the structure to the list of drawn structures
            x_drawn.append(x)
            n_drawn += 1
        return x_drawn

    def get_best_drawn_structure(
        self,
        x_drawn,
        steps=1000,
        max_unc=None,
        dtrust=None,
        **kwargs,
    ):
        "Get the best drawn structure from the random sampling."
        # Initialize the best energy and position
        self.best_energy = inf
        self.best_pos = None
        self.best_energy_no_crit = inf
        self.best_pos_no_crit = None
        # Calculate the energy of the initial structure if used
        if self.use_initial_struc:
            # Get the energy of the structure
            e = self.optimizable.get_potential_energy()
            # Check if the energy is lower than the best energy
            self.check_best_structure(
                e=e,
                pos=self.optimizable.get_positions(),
                max_unc=max_unc,
                dtrust=dtrust,
                **kwargs,
            )
        # Check each drawn structure
        for x in x_drawn:
            # Get the new positions of the adsorbate
            pos = self.get_new_positions(x, **kwargs)
            # Set the positions
            self.optimizable.set_positions(pos)
            # Check if the initial optimization is used
            if self.use_initial_opt:
                # Run the local optimization
                _, used_steps = self.local_optimize(
                    atoms=self.optimizable,
                    fmax=self.initial_fmax,
                    steps=self.initial_steps,
                    max_unc=max_unc,
                    dtrust=dtrust,
                    **kwargs,
                )
                steps -= used_steps
                self.steps += used_steps
                pos = self.optimizable.get_positions()
            else:
                steps -= 1
                self.steps += 1
            # Get the energy of the structure
            e = self.optimizable.get_potential_energy()
            # Check if the energy is lower than the best energy
            self.check_best_structure(
                e=e,
                pos=pos,
                max_unc=max_unc,
                dtrust=dtrust,
                **kwargs,
            )
        # Return the best position and number of steps
        if self.best_energy == inf:
            self.message(
                "Uncertainty or trust distance is above the maximum allowed."
            )
            return self.best_pos_no_crit, steps
        return self.best_pos, steps

    def check_best_structure(
        self,
        e,
        pos,
        max_unc=None,
        dtrust=None,
        **kwargs,
    ):
        "Check if the structure is the best one."
        # Check if the energy is lower than the best energy
        if e < self.best_energy:
            # Update the best energy and position without criteria
            if e < self.best_energy_no_crit:
                self.best_energy_no_crit = e
                self.best_pos_no_crit = pos.copy()
            # Check if criteria are met
            is_within_crit = True
            # Check if the uncertainty is above the maximum allowed
            if max_unc is not None:
                unc = self.get_uncertainty()
                if unc > max_unc:
                    is_within_crit = False
            # Check if the structures are within the trust distance
            if dtrust is not None:
                within_dtrust = self.is_within_dtrust(dtrust=dtrust)
                if not within_dtrust:
                    is_within_crit = False
            # Update the best energy and position
            if is_within_crit:
                self.best_energy = e
                self.best_pos = pos.copy()
        return self.best_energy, self.best_pos

    def rotation_matrix(self, angles, positions):
        "Rotate the adsorbate"
        # Get the angles
        theta1, theta2, theta3 = angles
        # Calculate the trigonometric functions
        cos1 = cos(theta1)
        sin1 = sin(theta1)
        cos2 = cos(theta2)
        sin2 = sin(theta2)
        cos3 = cos(theta3)
        sin3 = sin(theta3)
        # Calculate the full rotation matrix
        R = asarray(
            [
                [cos2 * cos3, cos2 * sin3, -sin2],
                [
                    sin1 * sin2 * cos3 - cos1 * sin3,
                    sin1 * sin2 * sin3 + cos1 * cos3,
                    sin1 * cos2,
                ],
                [
                    cos1 * sin2 * cos3 + sin1 * sin3,
                    cos1 * sin2 * sin3 - sin1 * cos3,
                    cos1 * cos2,
                ],
            ]
        )
        # Calculate the rotation of the positions
        positions = matmul(positions, R)
        return positions

    def get_new_positions(self, x, **kwargs):
        "Get the new positions of the adsorbate."
        # Get the positions
        pos = self.positions0.copy()
        # Calculate the positions of the adsorbate
        n_slab = self.n_slab
        n_all = self.n_slab + self.n_ads
        pos_ads = pos[n_slab:n_all]
        pos_ads = self.rotation_matrix(x[3:6], pos_ads)
        pos_ads += (self.cell * x[:3].reshape(-1, 1)).sum(axis=0)
        pos[n_slab:n_all] = pos_ads
        # Calculate the positions of the second adsorbate
        if self.n_ads2 > 0:
            pos_ads2 = pos[n_all:]
            pos_ads2 = self.rotation_matrix(x[9:12], pos_ads2)
            pos_ads2 += (self.cell * x[6:9].reshape(-1, 1)).sum(axis=0)
            pos[n_all:] = pos_ads2
        return pos

    def evaluate_value(self, x, atoms, **kwargs):
        "Evaluate the value of the adsorption."
        # Get the new positions of the adsorption
        pos = self.get_new_positions(x, **kwargs)
        # Set the positions
        atoms.set_positions(pos)
        # Get the potential energy
        return atoms.get_potential_energy()

    def get_reference_energy(self, **kwargs):
        "Get the reference energy of the structure."
        # If the repulsive check is not used, return 0.0
        if not self.use_repulsive_check:
            return 0.0
        # Calculate the energy of the isolated slab
        atoms = self.slab.copy()
        atoms.calc = self.repulsive_calculator
        e_ref = atoms.get_potential_energy()
        # Calculate the energy of the isolated adsorbate
        atoms = self.adsorbate.copy()
        atoms.calc = self.repulsive_calculator
        e_ref += atoms.get_potential_energy()
        # Calculate the energy of the isolated second adsorbate
        if self.adsorbate2 is not None:
            atoms = self.adsorbate2.copy()
            atoms.calc = self.repulsive_calculator
            e_ref += atoms.get_potential_energy()
        return e_ref

    def make_initial_structure(self, **kwargs):
        "Get the initial structure for the optimization."
        x_drawn = self.draw_random_structures(n_random_draws=1, **kwargs)
        x_drawn = x_drawn[0]
        pos = self.get_new_positions(x_drawn, **kwargs)
        self.optimizable.set_positions(pos)
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            slab=self.slab,
            adsorbate=self.adsorbate,
            adsorbate2=self.adsorbate2,
            bounds=self.bounds,
            n_random_draws=self.n_random_draws,
            use_initial_struc=self.use_initial_struc,
            use_initial_opt=self.use_initial_opt,
            initial_fmax=self.initial_fmax,
            initial_steps=self.initial_steps,
            use_repulsive_check=self.use_repulsive_check,
            repulsive_tol=self.repulsive_tol,
            repulsive_calculator=self.repulsive_calculator,
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
