from .method import OptimizerMethod
from ase.parallel import world
from ase.constraints import FixAtoms, FixBondLengths
import itertools
from numpy import array, asarray, concatenate, cos, matmul, pi, sin
from numpy.linalg import norm
from scipy import __version__ as scipy_version
from scipy.optimize import dual_annealing


class AdsorptionOptimizer(OptimizerMethod):
    """
    The AdsorptionOptimizer is used to run a global optimization of
    an adsorption on a surface.
    A single structure will be created and optimized.
    Simulated annealing will be used to global optimize the structure.
    The adsorbate is optimized on a surface, where the bond-lengths of the
    adsorbate atoms are fixed and the slab atoms are fixed.
    The AdsorptionOptimizer is applicable to be used with
    active learning.
    """

    def __init__(
        self,
        slab,
        adsorbate,
        adsorbate2=None,
        bounds=None,
        opt_kwargs={},
        bond_tol=1e-8,
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
                The boundary conditions used for the global optimization in
                form of the simulated annealing.
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.
            opt_kwargs: dict
                The keyword arguments for the simulated annealing optimization.
            bond_tol: float
                The bond tolerance used for the FixBondLengths.
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
        self.create_slab_ads(slab, adsorbate, adsorbate2, bond_tol=bond_tol)
        # Create the boundary conditions
        self.setup_bounds(bounds)
        # Set the parameters
        self.update_arguments(
            opt_kwargs=opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def get_structures(
        self,
        get_all=True,
        properties=[],
        allow_calculation=True,
        **kwargs,
    ):
        structures = self.copy_atoms(
            self.optimizable,
            properties=properties,
            allow_calculation=allow_calculation,
            **kwargs,
        )
        structures.set_constraint(self.constraints_org)
        return structures

    def create_slab_ads(
        self,
        slab,
        adsorbate,
        adsorbate2=None,
        bond_tol=1e-8,
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
            bond_tol: float
                The bond tolerance used for the FixBondLengths.

        Returns:
            self: object
                The object itself.
        """
        # Check the slab and adsorbate are given
        if slab is None or adsorbate is None:
            raise ValueError("The slab and adsorbate must be given!")
        # Save the bond length tolerance
        self.bond_tol = float(bond_tol)
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
        # Store the original constraints
        self.constraints_org = [c.copy() for c in optimizable.constraints]
        # Make constraints for optimization
        self.constraints_used = [FixAtoms(indices=list(range(self.n_slab)))]
        self.constraints_new = [FixAtoms(indices=list(range(self.n_slab)))]
        if self.n_ads > 1:
            # Get the fixed bond length pairs
            pairs = itertools.combinations(
                range(self.n_slab, self.n_slab + self.n_ads),
                2,
            )
            pairs = asarray(list(pairs))
            # Get the bond lengths
            bondlengths = norm(
                self.positions0[pairs[:, 0]] - self.positions0[pairs[:, 1]],
                axis=1,
            )
            # Add the constraints
            self.constraints_new.append(
                FixBondLengths(
                    pairs=pairs,
                    tolerance=self.bond_tol,
                    bondlengths=bondlengths,
                )
            )
        if self.n_ads2 > 1:
            # Get the fixed bond length pairs
            pairs = itertools.combinations(
                range(self.n_slab + self.n_ads, self.natoms),
                2,
            )
            pairs = asarray(list(pairs))
            # Get the bond lengths
            bondlengths = norm(
                self.positions0[pairs[:, 0]] - self.positions0[pairs[:, 1]],
                axis=1,
            )
            # Add the constraints
            self.constraints_new.append(
                FixBondLengths(
                    pairs=pairs,
                    tolerance=self.bond_tol,
                    bondlengths=bondlengths,
                )
            )
        optimizable.set_constraint(self.constraints_new)
        # Setup the optimizable structure
        self.setup_optimizable(optimizable)
        return self

    def setup_bounds(self, bounds=None):
        """
        Setup the boundary conditions for the global optimization.

        Parameters:
            bounds: (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for the global optimization in
                form of the simulated annealing.
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
        unc_convergence=None,
        **kwargs,
    ):
        # Check if the optimization can take any steps
        if steps <= 0:
            return self._converged
        # Use original constraints
        self.optimizable.set_constraint(self.constraints_used)
        # Perform the simulated annealing
        sol = dual_annealing(
            self.evaluate_value,
            bounds=self.bounds,
            maxfun=steps,
            **self.opt_kwargs,
        )
        # Set the positions
        self.evaluate_value(sol["x"])
        # Set the new constraints
        self.optimizable.set_constraint(self.constraints_new)
        # Calculate the maximum force to check convergence
        if fmax > self.get_fmax():
            # Check if the optimization is converged
            self._converged = self.check_convergence(
                converged=True,
                max_unc=max_unc,
                unc_convergence=unc_convergence,
            )
        return self._converged

    def is_energy_minimized(self):
        return True

    def is_parallel_allowed(self):
        return False

    def update_arguments(
        self,
        slab=None,
        adsorbate=None,
        adsorbate2=None,
        bounds=None,
        opt_kwargs=None,
        bond_tol=None,
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
                The boundary conditions used for the global optimization in
                form of the simulated annealing.
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.
            opt_kwargs: dict
                The keyword arguments for the simulated annealing optimization.
            bond_tol: float
                The bond tolerance used for the FixBondLengths.
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
        # Set the optimizer kwargs
        if opt_kwargs is not None:
            self.opt_kwargs = opt_kwargs.copy()
        if bond_tol is not None:
            self.bond_tol = float(bond_tol)
        # Set the parameters in the parent class
        super().update_arguments(
            optimizable=None,
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
                bond_tol=self.bond_tol,
            )
        # Create the boundary conditions
        if bounds is not None:
            self.setup_bounds(bounds)
        return self

    def set_seed(self, seed=None):
        super().set_seed(seed)
        # Set the seed for the random number generator
        if scipy_version < "1.15":
            self.opt_kwargs["seed"] = self.seed
        else:
            self.opt_kwargs["rng"] = self.rng
        return self

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

    def evaluate_value(self, x, **kwargs):
        "Evaluate the value of the adsorption."
        # Get the new positions of the adsorption
        pos = self.get_new_positions(x, **kwargs)
        # Set the positions
        self.optimizable.set_positions(pos)
        # Get the potential energy
        return self.optimizable.get_potential_energy()

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            slab=self.slab,
            adsorbate=self.adsorbate,
            adsorbate2=self.adsorbate2,
            bounds=self.bounds,
            opt_kwargs=self.opt_kwargs,
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
